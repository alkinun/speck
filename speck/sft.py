"""Prepare and load masked packed data for supervised instruction tuning."""

import hashlib
import json
import os
import shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download

from speck.chat import ChatFormatError
from speck.common import base_dir, dist_info
from speck.dataloader import manifest_fingerprint

FORMAT_VERSION = 3
default_sft_data_dir = Path(base_dir()) / "data" / f"SpeckChat1-v{FORMAT_VERSION}"


def resolve_sft_data_dir(config, output_dir=None):
    """Resolve an explicit path or derive an isolated cache from the dataset name."""

    if output_dir is not None:
        return Path(output_dir).expanduser()
    dataset_name = config["repo"].rsplit("/", 1)[-1]
    return Path(base_dir()) / "data" / f"{dataset_name}-v{FORMAT_VERSION}"


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_dataset_config(config):
    required = {"repo", "revision", "files", "expected_samples", "validation_samples"}
    if set(config) != required:
        missing = sorted(required - set(config))
        unknown = sorted(set(config) - required)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError("invalid SFT dataset config: " + "; ".join(details))
    if not isinstance(config["revision"], str) or len(config["revision"]) != 40:
        raise ValueError("dataset revision must be a full commit hash")
    if not isinstance(config["files"], list) or not config["files"]:
        raise ValueError("dataset files must be a non-empty list")
    expected = config["expected_samples"]
    validation = config["validation_samples"]
    if not isinstance(expected, int) or not isinstance(validation, int):
        raise ValueError("dataset sample counts must be integers")
    if expected < 2 or not 0 < validation < expected:
        raise ValueError(
            "dataset must contain at least two samples with a positive validation count below "
            "the total"
        )


def _truncate_conversation(tokens, mask, tokenizer, maximum):
    if len(tokens) <= maximum:
        return tokens, mask, False
    assistant_id = tokenizer.role_ids["assistant"]
    user_id = tokenizer.role_ids["user"]
    assistant_index = max(
        index for index, token in enumerate(tokens) if token == assistant_id and not mask[index]
    )
    user_index = max(
        index for index, token in enumerate(tokens[:assistant_index]) if token == user_id
    )
    newline = list(tokenizer.newline_ids)
    assistant = tokens[assistant_index:]
    assistant_mask = mask[assistant_index:]
    user_header = [tokenizer.bos_id, user_id]
    user_segment = tokens[user_index + 1 : assistant_index]
    available_user = maximum - len(user_header) - len(assistant)
    minimum_user = tokenizer.base.encode("\n...") + [tokenizer.eos_id] + newline
    if available_user >= len(minimum_user):
        if len(user_segment) > available_user:
            suffix_length = available_user - len(newline)
            user_segment = newline + user_segment[-suffix_length:]
        truncated_tokens = user_header + user_segment + assistant
        truncated_mask = [False] * (len(user_header) + len(user_segment))
        truncated_mask.extend(assistant_mask)
    else:
        assistant_header_length = 1 + len(newline)
        assistant_budget = maximum - len(user_header) - len(minimum_user)
        if assistant_budget <= assistant_header_length:
            raise ValueError("context length is too short for the chat template")
        truncated_tokens = user_header + minimum_user
        truncated_tokens.extend(assistant[:assistant_budget])
        truncated_mask = [False] * (len(user_header) + len(minimum_user))
        truncated_mask.extend(assistant_mask[:assistant_budget])
    if len(truncated_tokens) > maximum or not any(truncated_mask):
        raise ValueError("could not retain an assistant target while truncating a conversation")
    return truncated_tokens, truncated_mask, True


def prepare_sft_dataset(
    config,
    tokenizer,
    sequence_lengths,
    output_dir=None,
    restart=False,
):
    """Download and atomically publish length-bucketed, assistant-masked rows."""

    _validate_dataset_config(config)
    sequence_lengths = tuple(sequence_lengths)
    if (
        not sequence_lengths
        or tuple(sorted(set(sequence_lengths))) != sequence_lengths
        or any(not isinstance(length, int) or length < 1 for length in sequence_lengths)
    ):
        raise ValueError("SFT sequence lengths must be unique positive integers in ascending order")
    output_dir = resolve_sft_data_dir(config, output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_sft_manifest(output_dir)
        expected = {**config, "sequence_lengths": list(sequence_lengths)}
        if manifest["dataset"] != expected or manifest["tokenizer"] != tokenizer.metadata():
            raise ValueError("prepared SFT dataset does not match the configuration")
        return manifest
    if output_dir.exists():
        raise FileExistsError(f"incomplete SFT dataset exists: {output_dir}")

    building = output_dir.with_name(output_dir.name + ".building")
    if building.exists():
        if not restart:
            raise FileExistsError(f"incomplete SFT dataset build exists: {building}")
        shutil.rmtree(building)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()

    split_files = {
        split: {
            length: {
                "tokens": (building / f"{split}.{length}.tokens.bin").open("wb"),
                "mask": (building / f"{split}.{length}.mask.bin").open("wb"),
            }
            for length in sequence_lengths
        }
        for split in ("train", "val")
    }
    stats = {
        split: {
            "samples": 0,
            "input_samples": 0,
            "rejected_samples": 0,
            "rejection_reasons": Counter(),
            "tokens": 0,
            "supervised_tokens": 0,
            "truncated_samples": 0,
            "sources": Counter(),
            "buckets": {
                length: {
                    "samples": 0,
                    "tokens": 0,
                    "supervised_tokens": 0,
                    "truncated_samples": 0,
                }
                for length in sequence_lengths
            },
        }
        for split in split_files
    }
    sample_index = 0
    try:
        for filename in config["files"]:
            path = hf_hub_download(
                config["repo"],
                filename,
                revision=config["revision"],
                repo_type="dataset",
            )
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=["messages", "source"], batch_size=256):
                for row in batch.to_pylist():
                    split = "val" if sample_index < config["validation_samples"] else "train"
                    stats[split]["input_samples"] += 1
                    try:
                        tokens, mask = tokenizer.encode_messages(row["messages"])
                    except ChatFormatError as error:
                        stats[split]["rejected_samples"] += 1
                        stats[split]["rejection_reasons"][str(error)] += 1
                        sample_index += 1
                        continue
                    if not any(mask):
                        stats[split]["rejected_samples"] += 1
                        stats[split]["rejection_reasons"][
                            "conversation has no assistant target"
                        ] += 1
                        sample_index += 1
                        continue
                    tokens, mask, truncated = _truncate_conversation(
                        tokens,
                        mask,
                        tokenizer,
                        sequence_lengths[-1] + 1,
                    )
                    sequence_length = next(
                        length for length in sequence_lengths if len(tokens) <= length + 1
                    )
                    content_tokens = len(tokens)
                    padding = sequence_length + 1 - len(tokens)
                    tokens.extend([tokenizer.eos_id] * padding)
                    mask.extend([False] * padding)
                    split_files[split][sequence_length]["tokens"].write(
                        np.asarray(tokens, dtype="<u2").tobytes()
                    )
                    split_files[split][sequence_length]["mask"].write(
                        np.asarray(mask, dtype=np.uint8).tobytes()
                    )
                    supervised_tokens = sum(mask)
                    stats[split]["samples"] += 1
                    stats[split]["tokens"] += content_tokens
                    stats[split]["supervised_tokens"] += supervised_tokens
                    stats[split]["truncated_samples"] += int(truncated)
                    stats[split]["sources"][row["source"]] += 1
                    bucket = stats[split]["buckets"][sequence_length]
                    bucket["samples"] += 1
                    bucket["tokens"] += content_tokens
                    bucket["supervised_tokens"] += supervised_tokens
                    bucket["truncated_samples"] += int(truncated)
                    sample_index += 1
                    if sample_index % 10_000 == 0:
                        print(f"Serialized {sample_index:,} conversations")
        if sample_index != config["expected_samples"]:
            raise ValueError(
                f"expected {config['expected_samples']:,} samples, found {sample_index:,}"
            )
        for split, values in stats.items():
            if not values["samples"]:
                raise ValueError(f"SFT {split} split has no accepted samples")
        for buckets in split_files.values():
            for files in buckets.values():
                for handle in files.values():
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
    except BaseException:
        for buckets in split_files.values():
            for files in buckets.values():
                for handle in files.values():
                    if not handle.closed:
                        handle.close()
        raise

    splits = {}
    for split, values in stats.items():
        buckets = {}
        for length, bucket_stats in values["buckets"].items():
            token_path = building / f"{split}.{length}.tokens.bin"
            mask_path = building / f"{split}.{length}.mask.bin"
            buckets[str(length)] = {
                **bucket_stats,
                "sequence_length": length,
                "token_file": token_path.name,
                "token_sha256": _file_hash(token_path),
                "mask_file": mask_path.name,
                "mask_sha256": _file_hash(mask_path),
            }
        splits[split] = {
            "samples": values["samples"],
            "input_samples": values["input_samples"],
            "rejected_samples": values["rejected_samples"],
            "rejection_reasons": dict(sorted(values["rejection_reasons"].items())),
            "tokens": values["tokens"],
            "supervised_tokens": values["supervised_tokens"],
            "truncated_samples": values["truncated_samples"],
            "sources": dict(sorted(values["sources"].items())),
            "buckets": buckets,
        }
    manifest = {
        "format": "speck_sft",
        "format_version": FORMAT_VERSION,
        "dataset": {**config, "sequence_lengths": list(sequence_lengths)},
        "tokenizer": tokenizer.metadata(),
        "splits": splits,
    }
    _write_json(building / "manifest.json", manifest)
    os.replace(building, output_dir)
    return manifest


def load_sft_manifest(data_dir=None):
    path = Path(data_dir or default_sft_data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"SFT dataset is not prepared: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "speck_sft" or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SFT dataset manifest")
    return manifest


def verify_sft_dataset(data_dir, manifest, checksums=True):
    data_dir = Path(data_dir)
    for split, split_manifest in manifest["splits"].items():
        for bucket in split_manifest["buckets"].values():
            row_length = bucket["sequence_length"] + 1
            files = (
                ("token", np.dtype("<u2").itemsize),
                ("mask", np.dtype(np.uint8).itemsize),
            )
            for kind, itemsize in files:
                path = data_dir / bucket[f"{kind}_file"]
                expected_bytes = bucket["samples"] * row_length * itemsize
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise ValueError(f"invalid SFT {split} {kind} file: {path}")
                if checksums and _file_hash(path) != bucket[f"{kind}_sha256"]:
                    raise ValueError(
                        f"SFT {split} bucket {bucket['sequence_length']} {kind} checksum mismatch"
                    )


class SFTTokenStream:
    def __init__(self, data_dir, split, sequence_length, manifest):
        values = manifest["splits"][split]["buckets"][str(sequence_length)]
        self.samples = values["samples"]
        self.row_length = sequence_length + 1
        shape = (self.samples, self.row_length)
        self.tokens = np.memmap(
            Path(data_dir) / values["token_file"], mode="r", dtype="<u2", shape=shape
        )
        self.mask = np.memmap(
            Path(data_dir) / values["mask_file"], mode="r", dtype=np.uint8, shape=shape
        )
        if self.tokens.shape != self.mask.shape:
            raise ValueError(f"invalid packed SFT {split} stream")

    def read(self, start, count):
        end = start + count
        if start < 0 or end > self.samples:
            raise IndexError("packed SFT read is out of range")
        return (
            np.array(self.tokens[start:end], dtype=np.int64, copy=True),
            np.array(self.mask[start:end], dtype=np.bool_, copy=True),
        )

    def read_padded(self, start, count, pad_token_id):
        tokens = np.full((count, self.row_length), pad_token_id, dtype=np.int64)
        mask = np.zeros((count, self.row_length), dtype=np.bool_)
        valid = max(0, min(count, self.samples - start))
        if valid:
            token_values, mask_values = self.read(start, valid)
            tokens[:valid] = token_values
            mask[:valid] = mask_values
        return tokens, mask, valid


def _bucket_schedule(counts):
    events = [
        (index / count, sequence_length, index)
        for sequence_length, count in counts.items()
        for index in range(count)
    ]
    events.sort()
    return tuple((sequence_length, index) for _, sequence_length, index in events)


def sft_plan(manifest, split, device_tokens, world_size=1, accumulation=1):
    """Build one deterministic, proportionally interleaved bucket epoch."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if min(device_tokens, world_size, accumulation) < 1:
        raise ValueError("SFT bucket geometry must be positive")
    buckets = {}
    counts = {}
    for key, bucket in manifest["splits"][split]["buckets"].items():
        sequence_length = int(key)
        if device_tokens % sequence_length:
            raise ValueError("device tokens must be divisible by every SFT sequence length")
        batch_size = device_tokens // sequence_length
        global_batch_size = batch_size * world_size
        microbatches = (bucket["samples"] + global_batch_size - 1) // global_batch_size
        buckets[sequence_length] = {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "global_batch_size": global_batch_size,
            "available_samples": bucket["samples"],
            "microbatches": microbatches,
        }
        counts[sequence_length] = microbatches
    schedule = _bucket_schedule(counts)
    real_microbatches = len(schedule)
    if real_microbatches < 1:
        raise ValueError("SFT buckets do not contain one optimizer batch")
    dummy_microbatches = (-real_microbatches) % accumulation
    if dummy_microbatches:
        dummy_length = min(
            sequence_length
            for sequence_length, bucket in buckets.items()
            if bucket["available_samples"]
        )
        first_dummy = buckets[dummy_length]["microbatches"]
        schedule += tuple(
            (dummy_length, first_dummy + index) for index in range(dummy_microbatches)
        )
    cycle_microbatches = len(schedule)
    scheduled = Counter(sequence_length for sequence_length, _ in schedule)
    for sequence_length, bucket in buckets.items():
        bucket["scheduled_microbatches"] = scheduled[sequence_length]
        bucket["used_samples"] = bucket["available_samples"]
        bucket["padded_samples"] = (
            scheduled[sequence_length] * bucket["global_batch_size"] - bucket["available_samples"]
        )
    fingerprint = manifest_fingerprint(
        {
            "manifest": manifest_fingerprint(manifest),
            "split": split,
            "device_tokens": device_tokens,
            "world_size": world_size,
            "accumulation": accumulation,
            "schedule": schedule,
        }
    )
    return {
        "split": split,
        "device_tokens": device_tokens,
        "world_size": world_size,
        "accumulation": accumulation,
        "real_microbatches": real_microbatches,
        "dummy_microbatches": dummy_microbatches,
        "cycle_microbatches": cycle_microbatches,
        "context_tokens": cycle_microbatches * device_tokens * world_size,
        "buckets": buckets,
        "schedule": schedule,
        "fingerprint": fingerprint,
    }


def _loader_state(manifest, plan, consumed):
    epoch, position = divmod(consumed, plan["cycle_microbatches"])
    sequence_length, bucket_batch = plan["schedule"][position]
    return {
        "format_version": 2,
        "contract": "batch_start",
        "manifest": manifest_fingerprint(manifest),
        "plan": plan["fingerprint"],
        "split": plan["split"],
        "global_consumed_microbatches": consumed,
        "cycle_microbatches": plan["cycle_microbatches"],
        "schedule_position": position,
        "epoch": epoch,
        "sequence_length": sequence_length,
        "bucket_batch": bucket_batch,
        "batch_size": plan["buckets"][sequence_length]["batch_size"],
        "world_size": plan["world_size"],
    }


def sft_loader(
    tokenizer,
    device_tokens,
    accumulation=1,
    split="train",
    device="cuda",
    resume_state_dict=None,
    data_dir=None,
):
    """Yield a deterministic mix of isolated, length-bucketed chat rows."""

    data_dir = Path(data_dir or default_sft_data_dir)
    manifest = load_sft_manifest(data_dir)
    if manifest["tokenizer"] != tokenizer.metadata():
        raise ValueError("SFT dataset and tokenizer do not match")
    rank, _, world_size = dist_info()
    plan = sft_plan(manifest, split, device_tokens, world_size, accumulation)
    streams = {
        sequence_length: SFTTokenStream(data_dir, split, sequence_length, manifest)
        for sequence_length, bucket in plan["buckets"].items()
        if bucket["scheduled_microbatches"]
    }
    consumed = 0
    if resume_state_dict is not None:
        consumed = resume_state_dict.get("global_consumed_microbatches")
        if (
            not isinstance(consumed, int)
            or consumed < 0
            or resume_state_dict != _loader_state(manifest, plan, consumed)
        ):
            raise ValueError("invalid SFT loader resume state")
    device = torch.device(device)
    while True:
        state = _loader_state(manifest, plan, consumed)
        sequence_length = state["sequence_length"]
        batch_size = state["batch_size"]
        global_batch_size = plan["buckets"][sequence_length]["global_batch_size"]
        start = state["bucket_batch"] * global_batch_size + rank * batch_size
        stream = streams[sequence_length]
        token_values, mask_values, _ = stream.read_padded(start, batch_size, tokenizer.eos_id)
        token_rows = torch.from_numpy(token_values)
        mask_rows = torch.from_numpy(mask_values)
        inputs = token_rows[:, :-1].clone()
        targets = token_rows[:, 1:].clone()
        targets.masked_fill_(~mask_rows[:, 1:], -100)
        if device.type == "cuda":
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        yield inputs, targets, state
        consumed += 1


def sft_optimization_step(
    train_model,
    parameters,
    optimizer,
    loader,
    batch,
    accumulation,
    grad_clip,
    lr,
    distributed=False,
):
    """Optimize one globally assistant-token-normalized SFT batch."""

    batches = [batch]
    for _ in range(accumulation - 1):
        batches.append(next(loader))
    next_batch = next(loader)
    device = batch[0].device
    supervised = torch.tensor(
        sum(int((current[1] != -100).sum()) for current in batches),
        device=device,
        dtype=torch.long,
    )
    if distributed:
        dist.all_reduce(supervised, op=dist.ReduceOp.SUM)
    if not supervised.item():
        raise ValueError("SFT optimizer batch has no supervised tokens")

    optimizer.zero_grad(set_to_none=True)
    local_loss = torch.zeros((), device=device)
    world_size = dist.get_world_size() if distributed else 1
    scale = world_size / supervised.item()
    for index, current in enumerate(batches):
        context = (
            train_model.no_sync() if distributed and index + 1 < accumulation else nullcontext()
        )
        with context:
            loss = train_model(current[0], current[1], loss_reduction="sum")
            (loss * scale).backward()
        local_loss += loss.detach()
    if distributed:
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    if not torch.isfinite(local_loss).item():
        raise FloatingPointError("non-finite SFT training loss")
    for group in optimizer.param_groups:
        group["lr"] = lr
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return local_loss / supervised, grad_norm, next_batch, int(supervised.item())


@torch.no_grad()
def validate_sft(model, loader, steps, distributed=False):
    training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        loss = torch.zeros((), device=device)
        supervised = torch.zeros((), device=device, dtype=torch.long)
        for _ in range(steps):
            inputs, targets, _ = next(loader)
            loss += model(inputs, targets, loss_reduction="sum")
            supervised += (targets != -100).sum()
        if distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(supervised, op=dist.ReduceOp.SUM)
        if not supervised.item():
            raise ValueError("SFT validation batch has no supervised tokens")
        mean_loss = loss / supervised
        if not torch.isfinite(mean_loss).item():
            raise FloatingPointError("non-finite SFT validation loss")
        return mean_loss.item(), int(supervised.item())
    finally:
        model.train(training)
