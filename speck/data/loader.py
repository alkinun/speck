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

from speck.data.dataset import default_data_dir, load_manifest
from speck.operations.runtime import dist_info

_MAX_WEIGHT_CYCLE = 100_000


class PackedTokenSource:
    """Expose one source and split as a contiguous memory-mapped token stream.

    Row-packed sources carry a parallel uint8 loss mask over the same positions.
    """

    def __init__(self, data_dir, source, split):
        self.data_dir = Path(data_dir)
        self.source_id = source["id"]
        split_manifest = source["splits"][split]
        self.row_tokens = (source.get("packing") or {}).get("row_tokens")
        self.shards, self.ends = self._map(split_manifest["shards"], "<u2")
        self.masks = None
        if self.row_tokens is not None:
            self.masks, mask_ends = self._map(split_manifest["mask_shards"], "u1")
            if mask_ends != self.ends:
                raise ValueError(f"mask shards do not align with tokens for {self.source_id}")
        if not self.shards or self.ends[-1] != split_manifest["tokens"]:
            raise ValueError(f"invalid {split} token count for source {self.source_id}")
        self.total_tokens = self.ends[-1]

    def _map(self, shards, dtype):
        arrays, ends, total = [], [], 0
        for shard in shards:
            path = self.data_dir / shard["path"]
            if (
                not path.exists()
                or path.stat().st_size != shard["tokens"] * np.dtype(dtype).itemsize
            ):
                raise ValueError(f"invalid packed shard: {path}")
            arrays.append(np.memmap(path, mode="r", dtype=dtype))
            total += shard["tokens"]
            ends.append(total)
        return arrays, ends

    def read(self, start, count, dtype=np.int64, mask=False):
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError(f"packed token read is out of range for source {self.source_id}")
        arrays = self.masks if mask else self.shards
        if count == 0:
            return np.empty(0, dtype=dtype)
        pieces = []
        position = start
        remaining = count
        while remaining:
            shard_index = bisect.bisect_right(self.ends, position)
            shard_start = 0 if shard_index == 0 else self.ends[shard_index - 1]
            offset = position - shard_start
            take = min(remaining, len(arrays[shard_index]) - offset)
            pieces.append(arrays[shard_index][offset : offset + take])
            position += take
            remaining -= take
        return np.array(
            pieces[0] if len(pieces) == 1 else np.concatenate(pieces),
            dtype=dtype,
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
    if Counter(cycle) != Counter(dict(zip(source_ids, weights, strict=False))):
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
    counts = dict.fromkeys(source_ids, 0)
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


def _sequence_schedule(manifest, split):
    """Return the opt-in per-sequence schedule, or None for per-microbatch selection.

    A sequence schedule selects a source for every training sequence, so the data order
    is independent of device batch size and world size. Validation keeps its
    per-microbatch round-robin, which gives each validation batch one source for the
    per-source validation losses.
    """

    return manifest["mixture"].get("schedule") if split == "train" else None


def _schedule_stride(manifest, split, sequence_length, global_stride):
    """Return the token span of one scheduling unit for this manifest and split."""

    schedule = _sequence_schedule(manifest, split)
    if schedule is None:
        return global_stride
    if sequence_length != schedule["sequence_length"]:
        raise ValueError(
            f"packed dataset schedules {schedule['sequence_length']}-token sequences; "
            f"train it at that sequence length, not {sequence_length}"
        )
    return sequence_length


def sequence_schedule(manifest, first_sequence, count):
    """Return the source and source cursor of consecutive global training sequences.

    Global sequence g starts at global token g * sequence_length; a source's cursor
    counts the sequences scheduled to it before g.
    """

    sequence_length = manifest["mixture"]["schedule"]["sequence_length"]
    cursors = source_selection_counts(
        manifest, "train", first_sequence * sequence_length, sequence_length
    )
    rows = []
    for sequence in range(first_sequence, first_sequence + count):
        source_id, _ = scheduled_source(
            manifest, "train", sequence * sequence_length, sequence_length
        )
        rows.append((source_id, cursors[source_id]))
        cursors[source_id] += 1
    return rows


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


def _validate_geometry(sequence_length, batch_size, world_size):
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (sequence_length, batch_size, world_size)
    ):
        raise ValueError("loader geometry must contain positive integers")


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
    _validate_geometry(sequence_length, batch_size, world_size)
    global_stride = sequence_length * batch_size * world_size
    if global_consumed_tokens < 0 or global_consumed_tokens % global_stride:
        raise ValueError("loader offset must align with distributed microbatches")
    stride = _schedule_stride(manifest, split, sequence_length, global_stride)
    return _loader_state(
        manifest,
        manifest_fingerprint(manifest),
        split,
        global_consumed_tokens,
        stride,
        (sequence_length, batch_size, world_size),
    )


def _loader_state(manifest, dataset_hash, split, global_consumed_tokens, stride, geometry):
    selected_source, phase = scheduled_source(manifest, split, global_consumed_tokens, stride)
    counts = source_selection_counts(manifest, split, global_consumed_tokens, stride)
    sources = _source_map(manifest)
    offsets = {}
    epochs = {}
    schedule_end = manifest["requested_train_tokens"]
    for source_id, count in counts.items():
        total = sources[source_id]["splits"][split]["tokens"]
        batches_per_epoch = (total - 1) // stride
        if batches_per_epoch < 1:
            raise ValueError(
                f"packed source {source_id} is smaller than one distributed microbatch"
            )
        if split == "val" or global_consumed_tokens >= schedule_end:
            epoch, batch_offset = divmod(count, batches_per_epoch)
        else:
            epoch, batch_offset = 0, count
        offsets[source_id] = batch_offset * stride
        epochs[source_id] = epoch
    sequence_length, batch_size, world_size = geometry
    return {
        "format_version": 2,
        "contract": "batch_start",
        "manifest": dataset_hash,
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
    # A sequence schedule's cursors depend only on the global token offset, so it may
    # resume under another device batch size or world size at an aligned offset.
    sequenced = _sequence_schedule(manifest, split) is not None
    if state.get("sequence_length") != sequence_length or (
        not sequenced and state.get("batch_size") != batch_size
    ):
        raise ValueError("cannot resume with different batch geometry")
    if not sequenced and state.get("world_size") != world_size:
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


def _sequence_batches(
    manifest, dataset_hash, packed, global_consumed_tokens, rank, geometry, device
):
    """Yield microbatches whose rows are consecutive, individually scheduled sequences.

    Rank r reads global sequences first + r * batch_size + row, so every optimizer
    step trains on its next batch_tokens / sequence_length global sequences however
    they are split across ranks and accumulation microbatches.
    """

    sequence_length, batch_size, world_size = geometry
    global_stride = sequence_length * batch_size * world_size
    schedule_end = manifest["requested_train_tokens"]
    read_dtype = np.uint16 if device.type == "cuda" else np.int64
    while True:
        state = _loader_state(
            manifest, dataset_hash, "train", global_consumed_tokens, sequence_length, geometry
        )
        first = global_consumed_tokens // sequence_length + rank * batch_size
        tokens, masks = [], []
        for sequence, (source_id, cursor) in enumerate(
            sequence_schedule(manifest, first, batch_size), first
        ):
            source = packed[source_id]
            sequences_per_epoch = (source.total_tokens - 1) // sequence_length
            if cursor >= sequences_per_epoch and sequence * sequence_length < schedule_end:
                raise RuntimeError(
                    f"packed source {source_id} exhausted at global sequence "
                    f"{sequence:,}; the configured mixture cannot be preserved"
                )
            offset = cursor % sequences_per_epoch * sequence_length
            tokens.append(source.read(offset, sequence_length + 1, dtype=read_dtype))
            if source.masks is not None:
                masks.append(source.read(offset, sequence_length + 1, dtype=np.uint8, mask=True))
            else:
                masks.append(np.ones(sequence_length + 1, dtype=np.uint8))
        rows = torch.from_numpy(np.stack(tokens))
        if device.type == "cuda":
            rows = rows.pin_memory().to(device, dtype=torch.int64, non_blocking=True)
        else:
            rows = rows.to(device)
        inputs = rows[:, :-1]
        targets = rows[:, 1:]
        if any(source.masks is not None for source in packed.values()):
            mask = torch.from_numpy(np.stack(masks)).to(device, non_blocking=True)
            targets = targets.masked_fill(mask[:, 1:] == 0, -100)
        yield inputs, targets, state
        global_consumed_tokens += global_stride


def packed_loader(
    tokenizer,
    batch_size,
    sequence_length,
    split="train",
    device: str | torch.device = "cuda",
    resume_state_dict=None,
    data_dir=None,
    initial_token_offset=0,
):
    """Yield microbatches and their exact batch-start cursor state.

    Microbatches hold one source unless the manifest opts into a sequence schedule.
    """

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    data_dir = Path(data_dir or default_data_dir / "packed")
    rank, _, world_size = dist_info()
    _validate_geometry(sequence_length, batch_size, world_size)
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
    for source in packed.values():
        if source.row_tokens not in (None, sequence_length):
            raise ValueError(
                f"source {source.source_id} is packed in {source.row_tokens}-token rows; "
                f"train it at that sequence length, not {sequence_length}"
            )
    local_stride = batch_size * sequence_length
    global_stride = local_stride * world_size
    required = local_stride + 1
    stride = _schedule_stride(manifest, split, sequence_length, global_stride)
    if split == "train":
        _validate_training_capacity(manifest, stride)
    if resume_state_dict is not None and initial_token_offset:
        raise ValueError("initial token offset cannot be combined with resume state")
    if split != "train" and initial_token_offset:
        raise ValueError("initial token offset is supported only for training")
    if resume_state_dict is None:
        initial = loader_state_for_offset(
            manifest,
            split,
            initial_token_offset,
            sequence_length,
            batch_size,
            world_size,
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
    if _sequence_schedule(manifest, split) is not None:
        yield from _sequence_batches(
            manifest,
            dataset_hash,
            packed,
            global_consumed_tokens,
            rank,
            (sequence_length, batch_size, world_size),
            device,
        )
        return

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
        read_dtype = np.uint16 if device.type == "cuda" else np.int64
        flat = torch.from_numpy(source.read(rank_offset, required, dtype=read_dtype))
        if device.type == "cuda":
            flat = flat.pin_memory().to(device, dtype=torch.int64, non_blocking=True)
        else:
            flat = flat.to(device)
        rows = flat.unfold(0, sequence_length + 1, sequence_length)
        inputs = rows[:, :-1]
        targets = rows[:, 1:]
        if source.masks is not None:
            mask = torch.from_numpy(source.read(rank_offset, required, dtype=np.uint8, mask=True))
            mask = mask.to(device, non_blocking=True).unfold(
                0, sequence_length + 1, sequence_length
            )
            targets = targets.masked_fill(mask[:, 1:] == 0, -100)
        yield inputs, targets, state

        source_offsets[source_id] += global_stride
        global_consumed_tokens += global_stride
        if split == "val" or global_consumed_tokens >= schedule_end:
            for wrapped_id, wrapped_source in packed.items():
                if source_offsets[wrapped_id] + global_stride + 1 > wrapped_source.total_tokens:
                    source_offsets[wrapped_id] = 0
                    source_epochs[wrapped_id] += 1
