"""Load deterministic distributed batches from source-separated packed shards."""

import bisect
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from speck.common import dist_info
from speck.dataset import default_data_dir, load_manifest

_MAX_WEIGHT_CYCLE = 100_000


class PackedTokenSource:
    """Expose one source and split as a contiguous memory-mapped token stream."""

    def __init__(self, data_dir, source, split):
        self.data_dir = Path(data_dir)
        self.source_id = source["id"]
        split_manifest = source["splits"][split]
        self.shard_manifests = split_manifest["shards"]
        self.shards = []
        self.ends = []
        total = 0
        for shard in self.shard_manifests:
            path = self.data_dir / shard["path"]
            expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
            if not path.exists() or path.stat().st_size != expected_bytes:
                raise ValueError(f"invalid packed token shard: {path}")
            self.shards.append(np.memmap(path, mode="r", dtype="<u2"))
            total += shard["tokens"]
            self.ends.append(total)
        if total != split_manifest["tokens"] or not self.shards:
            raise ValueError(f"invalid {split} token count for source {self.source_id}")
        self.total_tokens = total

    def read(self, start, count):
        if start < 0 or start + count > self.total_tokens:
            raise IndexError(f"packed token read is out of range for source {self.source_id}")
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
            pieces[0] if len(pieces) == 1 else np.concatenate(pieces),
            dtype=np.int64,
            copy=True,
        )


def manifest_fingerprint(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_map(manifest):
    return {source["id"]: source for source in manifest["sources"]}


@lru_cache(maxsize=None)
def _smooth_cycle(weight_items):
    """Build one exact smooth weighted round-robin cycle."""

    source_ids = tuple(source_id for source_id, _ in weight_items)
    fractions = tuple(Fraction(str(weight)) for _, weight in weight_items)
    scale = math.lcm(*(weight.denominator for weight in fractions))
    weights = tuple(int(weight * scale) for weight in fractions)
    divisor = math.gcd(*weights)
    weights = tuple(weight // divisor for weight in weights)
    total_weight = sum(weights)
    if total_weight > _MAX_WEIGHT_CYCLE:
        raise ValueError(f"exact mixture scheduling cycle exceeds {_MAX_WEIGHT_CYCLE:,} batches")
    current = [0] * len(source_ids)
    cycle = []
    for _ in range(total_weight):
        for index, weight in enumerate(weights):
            current[index] += weight
        selected = max(range(len(source_ids)), key=current.__getitem__)
        current[selected] -= total_weight
        cycle.append(source_ids[selected])
    if Counter(cycle) != Counter(dict(zip(source_ids, weights))):
        raise ValueError("mixture weights did not produce an exact scheduling cycle")
    return tuple(cycle)


def _phase_context(manifest, global_consumed_tokens, global_stride):
    phases = manifest["mixture"]["phases"]
    phase_start = 0
    phase_index = len(phases) - 1
    for index, phase in enumerate(phases):
        if global_consumed_tokens < phase["end_tokens"] or index + 1 == len(phases):
            phase_index = index
            break
        phase_start = phase["end_tokens"]
    first_batch = (phase_start + global_stride - 1) // global_stride
    batch_index = global_consumed_tokens // global_stride
    return phase_index, max(0, batch_index - first_batch)


def scheduled_source(manifest, split, global_consumed_tokens, global_stride):
    """Select a source using only the manifest and absolute global batch position."""

    source_ids = tuple(source["id"] for source in manifest["sources"])
    if split == "val":
        batch_index = global_consumed_tokens // global_stride
        return source_ids[batch_index % len(source_ids)], "validation"
    phase_index, phase_batch = _phase_context(manifest, global_consumed_tokens, global_stride)
    weights = manifest["mixture"]["phases"][phase_index]["weights"]
    cycle = _smooth_cycle(tuple((source_id, weights[source_id]) for source_id in source_ids))
    return cycle[phase_batch % len(cycle)], phase_index


def _add_cycle_counts(counts, cycle, batches):
    full_cycles, remainder = divmod(batches, len(cycle))
    cycle_counts = Counter(cycle)
    for source_id in counts:
        counts[source_id] += full_cycles * cycle_counts[source_id]
    for source_id, count in Counter(cycle[:remainder]).items():
        counts[source_id] += count


def source_selection_counts(manifest, split, global_consumed_tokens, global_stride):
    """Count source selections strictly before an aligned global position."""

    source_ids = tuple(source["id"] for source in manifest["sources"])
    counts = {source_id: 0 for source_id in source_ids}
    batches_before = global_consumed_tokens // global_stride
    if split == "val":
        full, remainder = divmod(batches_before, len(source_ids))
        for index, source_id in enumerate(source_ids):
            counts[source_id] = full + (index < remainder)
        return counts

    phases = manifest["mixture"]["phases"]
    phase_start = 0
    for phase_index, phase in enumerate(phases):
        first_batch = (phase_start + global_stride - 1) // global_stride
        final_batch = (phase["end_tokens"] + global_stride - 1) // global_stride
        if phase_index + 1 == len(phases) and batches_before > final_batch:
            final_batch = batches_before
        selected_batches = max(0, min(batches_before, final_batch) - first_batch)
        if selected_batches:
            weights = phase["weights"]
            cycle = _smooth_cycle(
                tuple((source_id, weights[source_id]) for source_id in source_ids)
            )
            _add_cycle_counts(counts, cycle, selected_batches)
        if batches_before <= final_batch:
            break
        phase_start = phase["end_tokens"]
    return counts


def _shard_diagnostic(source, split, source_offset):
    shards = source["splits"][split]["shards"]
    total = 0
    for index, shard in enumerate(shards):
        total += shard["tokens"]
        if source_offset < total:
            return {
                "source_id": source["id"],
                "index": index,
                "path": shard["path"],
                "source_offset": source_offset,
            }
    last = shards[-1]
    return {
        "source_id": source["id"],
        "index": len(shards) - 1,
        "path": last["path"],
        "source_offset": source_offset,
    }


def loader_state_for_offset(
    manifest,
    split,
    global_consumed_tokens,
    sequence_length,
    batch_size,
    world_size=1,
):
    """Construct the v2 batch-start state for an absolute global token offset."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if min(sequence_length, batch_size, world_size) < 1:
        raise ValueError("loader geometry must be positive")
    global_stride = sequence_length * batch_size * world_size
    if global_consumed_tokens < 0 or global_consumed_tokens % global_stride:
        raise ValueError("loader offset must align with distributed microbatches")
    selected_source, phase = scheduled_source(
        manifest, split, global_consumed_tokens, global_stride
    )
    counts = source_selection_counts(manifest, split, global_consumed_tokens, global_stride)
    sources = _source_map(manifest)
    offsets = {}
    epochs = {}
    schedule_end = manifest["requested_train_tokens"]
    for source_id, count in counts.items():
        total = sources[source_id]["splits"][split]["tokens"]
        batches_per_epoch = (total - 1) // global_stride
        if batches_per_epoch < 1:
            raise ValueError(
                f"packed source {source_id} is smaller than one distributed microbatch"
            )
        if split == "val":
            epoch, batch_offset = divmod(count, batches_per_epoch)
        elif global_consumed_tokens >= schedule_end:
            epoch, batch_offset = divmod(count, batches_per_epoch)
        else:
            epoch, batch_offset = 0, count
        offsets[source_id] = batch_offset * global_stride
        epochs[source_id] = epoch
    return {
        "format_version": 2,
        "contract": "batch_start",
        "manifest": manifest_fingerprint(manifest),
        "split": split,
        "global_consumed_tokens": global_consumed_tokens,
        "source_offsets": offsets,
        "source_epochs": epochs,
        "selected_source": selected_source,
        "phase": phase,
        "shard": _shard_diagnostic(sources[selected_source], split, offsets[selected_source]),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "world_size": world_size,
    }


def _validate_resume_state(
    state,
    manifest,
    split,
    sequence_length,
    batch_size,
    world_size,
):
    if not isinstance(state, dict) or state.get("format_version") != 2:
        raise ValueError("unsupported packed loader state format")
    if state.get("manifest") != manifest_fingerprint(manifest):
        raise ValueError("cannot resume with a different packed dataset")
    if state.get("split") != split:
        raise ValueError("cannot resume a different packed split")
    if state.get("sequence_length") != sequence_length or state.get("batch_size") != batch_size:
        raise ValueError("cannot resume with different batch geometry")
    if state.get("world_size") != world_size:
        raise ValueError("cannot resume with a different world size")
    offset = state.get("global_consumed_tokens")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("packed loader state has an invalid global token offset")
    expected = loader_state_for_offset(
        manifest,
        split,
        offset,
        sequence_length,
        batch_size,
        world_size,
    )
    for key in (
        "contract",
        "global_consumed_tokens",
        "source_offsets",
        "source_epochs",
        "selected_source",
        "phase",
        "shard",
    ):
        if state.get(key) != expected[key]:
            raise ValueError(f"packed loader state has an invalid {key}")
    return expected


def _validate_training_capacity(manifest, global_stride):
    final_offset = (
        (manifest["requested_train_tokens"] + global_stride - 1) // global_stride
    ) * global_stride
    counts = source_selection_counts(manifest, "train", final_offset, global_stride)
    sources = _source_map(manifest)
    for source_id, batches in counts.items():
        required = batches * global_stride + (1 if batches else 0)
        available = sources[source_id]["splits"]["train"]["tokens"]
        if required > available:
            raise ValueError(
                f"packed source {source_id} is too small for the training schedule: "
                f"needs {required:,} tokens, has {available:,}"
            )


def packed_loader(
    tokenizer,
    batch_size,
    sequence_length,
    split="train",
    device: str | torch.device = "cuda",
    resume_state_dict=None,
    data_dir=None,
):
    """Yield one-source microbatches and their exact batch-start cursor state."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest = load_manifest(data_dir)
    tokenizer_manifest = manifest["tokenizer"]
    if tokenizer.vocab_size != tokenizer_manifest["vocab_size"]:
        raise ValueError("packed dataset vocabulary does not match tokenizer")
    if tokenizer.fingerprint() != tokenizer_manifest["fingerprint"]:
        raise ValueError("packed dataset was created with a different tokenizer")

    sources = _source_map(manifest)
    packed = {
        source_id: PackedTokenSource(data_dir, source, split)
        for source_id, source in sources.items()
    }
    rank, _, world_size = dist_info()
    local_stride = batch_size * sequence_length
    global_stride = local_stride * world_size
    required = local_stride + 1
    if split == "train":
        _validate_training_capacity(manifest, global_stride)
    if resume_state_dict is None:
        initial = loader_state_for_offset(
            manifest, split, 0, sequence_length, batch_size, world_size
        )
    else:
        initial = _validate_resume_state(
            resume_state_dict,
            manifest,
            split,
            sequence_length,
            batch_size,
            world_size,
        )
    global_consumed_tokens = initial["global_consumed_tokens"]
    source_offsets = dict(initial["source_offsets"])
    source_epochs = dict(initial["source_epochs"])
    schedule_end = manifest["requested_train_tokens"]
    dataset_hash = manifest_fingerprint(manifest)
    device = torch.device(device)

    while True:
        source_id, phase = scheduled_source(manifest, split, global_consumed_tokens, global_stride)
        source = packed[source_id]
        source_offset = source_offsets[source_id]
        if source_offset + global_stride + 1 > source.total_tokens:
            if split == "train" and global_consumed_tokens < schedule_end:
                raise RuntimeError(
                    f"packed source {source_id} exhausted at global token "
                    f"{global_consumed_tokens:,}; the configured mixture cannot be preserved"
                )
            source_offset = 0
            source_offsets[source_id] = 0
            source_epochs[source_id] += 1
        if global_stride + 1 > source.total_tokens:
            raise ValueError(
                f"packed source {source_id} is smaller than one distributed microbatch"
            )

        state = {
            "format_version": 2,
            "contract": "batch_start",
            "manifest": dataset_hash,
            "split": split,
            "global_consumed_tokens": global_consumed_tokens,
            "source_offsets": dict(source_offsets),
            "source_epochs": dict(source_epochs),
            "selected_source": source_id,
            "phase": phase,
            "shard": _shard_diagnostic(sources[source_id], split, source_offset),
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        rank_offset = source_offset + rank * local_stride
        flat = torch.from_numpy(source.read(rank_offset, required))
        if device.type == "cuda":
            flat = flat.pin_memory().to(device, non_blocking=True)
        else:
            flat = flat.to(device)
        rows = flat.unfold(0, sequence_length + 1, sequence_length)
        inputs = rows[:, :-1]
        targets = rows[:, 1:]
        yield inputs, targets, state

        source_offsets[source_id] += global_stride
        global_consumed_tokens += global_stride
        if split == "val" or global_consumed_tokens >= schedule_end:
            for wrapped_id, wrapped_source in packed.items():
                if source_offsets[wrapped_id] + global_stride + 1 > wrapped_source.total_tokens:
                    source_offsets[wrapped_id] = 0
                    source_epochs[wrapped_id] += 1
