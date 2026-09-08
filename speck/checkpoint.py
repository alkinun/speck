"""Provide atomic training checkpoints."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import torch

from speck.io import file_sha256


def _validate_step(step):
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return step


def _flush_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _flush_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def _transaction_paths(directory, step, transaction):
    paths = {
        "model": directory / f"model_{step:06d}.pt",
        "optimizer": directory / f"optimizer_{step:06d}.pt",
        "metadata": directory / f"metadata_{step:06d}.json",
        "timing": directory / f"timing_{step:06d}.json",
        "complete": directory / f"complete_{step:06d}",
    }
    suffix = f".{transaction}"
    staged = {name: path.with_name(path.name + ".tmp" + suffix) for name, path in paths.items()}
    backups = {name: path.with_name(path.name + ".backup" + suffix) for name, path in paths.items()}
    journal = directory / f".checkpoint-transaction-{step:06d}-{transaction}.json"
    return paths, staged, backups, journal


def _recover_transactions(directory):
    """Recover an interrupted same-step publication before checkpoint discovery."""

    directory = Path(directory)
    if not directory.is_dir():
        return
    pattern = re.compile(r"\.checkpoint-transaction-(\d{6})-([0-9]+-[0-9a-f]{32})\.json")
    for journal in sorted(directory.glob(".checkpoint-transaction-*.json")):
        match = pattern.fullmatch(journal.name)
        if match is None:
            raise RuntimeError(f"invalid checkpoint transaction journal: {journal.name}")
        step = int(match.group(1))
        transaction = match.group(2)
        record = json.loads(journal.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "format_version", "step", "transaction", "existing"}
            or record["format"] != "speck_checkpoint_transaction"
            or record["format_version"] != 1
            or record["step"] != step
            or record["transaction"] != transaction
            or not isinstance(record["existing"], list)
            or any(
                name not in {"model", "optimizer", "metadata", "timing", "complete"}
                for name in record["existing"]
            )
            or len(record["existing"]) != len(set(record["existing"]))
        ):
            raise RuntimeError(f"invalid checkpoint transaction record: {journal.name}")
        paths, staged, backups, _ = _transaction_paths(directory, step, transaction)
        existing = set(record["existing"])
        if not paths["complete"].exists():
            for name, path in paths.items():
                backup = backups[name]
                if backup.exists():
                    path.unlink(missing_ok=True)
                    os.replace(backup, path)
                elif name not in existing:
                    path.unlink(missing_ok=True)
        for temporary in (*staged.values(), *backups.values()):
            temporary.unlink(missing_ok=True)
        journal.unlink()
        _flush_directory(directory)


def save(directory, step, model, optimizer, metadata, timing=None):
    step = _validate_step(step)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _recover_transactions(directory)
    transaction = f"{os.getpid()}-{uuid.uuid4().hex}"
    paths, staged, backups, journal = _transaction_paths(directory, step, transaction)
    backed_up = []
    published = []
    rollback_complete = False
    try:
        torch.save(model, staged["model"])
        torch.save(optimizer, staged["optimizer"])
        _write_json(staged["metadata"], metadata)
        timing = timing() if callable(timing) else timing
        if timing is not None:
            _write_json(staged["timing"], timing)
        staged["complete"].write_text("complete\n", encoding="utf-8")
        for name in ("model", "optimizer", "complete"):
            _flush_file(staged[name])
        existing = [name for name, path in paths.items() if path.exists()]
        journal_staging = journal.with_suffix(".json.tmp")
        _write_json(
            journal_staging,
            {
                "format": "speck_checkpoint_transaction",
                "format_version": 1,
                "step": step,
                "transaction": transaction,
                "existing": existing,
            },
        )
        os.replace(journal_staging, journal)
        _flush_directory(directory)

        # Hide the predecessor before moving any of its payloads. Readers either
        # observe the old complete checkpoint, an incomplete step during this
        # short publication window, or the fully published replacement.
        for name in ("complete", "model", "optimizer", "metadata", "timing"):
            if paths[name].exists():
                os.replace(paths[name], backups[name])
                backed_up.append(name)
        for name in ("model", "optimizer", "metadata"):
            os.replace(staged[name], paths[name])
            published.append(name)
        if timing is not None:
            os.replace(staged["timing"], paths["timing"])
            published.append("timing")
        os.replace(staged["complete"], paths["complete"])
        published.append("complete")
        _flush_directory(directory)
    except Exception:
        rollback_errors = []
        for name in reversed(published):
            try:
                paths[name].unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(error)
        for name in (*reversed([name for name in backed_up if name != "complete"]), "complete"):
            if name not in backed_up:
                continue
            try:
                os.replace(backups[name], paths[name])
            except OSError as error:
                rollback_errors.append(error)
        try:
            _flush_directory(directory)
        except OSError as error:
            rollback_errors.append(error)
        rollback_complete = not rollback_errors
        if rollback_errors:
            raise RuntimeError(
                "checkpoint publication failed and predecessor rollback was incomplete"
            )
        raise
    else:
        rollback_complete = True
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        _flush_directory(directory)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        if rollback_complete:
            for backup in backups.values():
                backup.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            journal.with_suffix(".json.tmp").unlink(missing_ok=True)


def completed_steps(directory):
    if not os.path.isdir(directory):
        return []
    _recover_transactions(directory)
    steps = []
    for name in os.listdir(directory):
        match = re.fullmatch(r"complete_(\d+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        if name == f"complete_{step:06d}":
            steps.append(step)
    return sorted(steps)


def latest(directory):
    steps = completed_steps(directory)
    return steps[-1] if steps else None


def prune(directory, keep):
    """Keep only the newest completed checkpoints in one run directory."""

    if not isinstance(keep, int) or keep < 1:
        raise ValueError("checkpoint retention must be a positive integer")
    for step in completed_steps(directory)[:-keep]:
        names = (
            f"complete_{step:06d}",
            f"model_{step:06d}.pt",
            f"optimizer_{step:06d}.pt",
            f"metadata_{step:06d}.json",
            f"timing_{step:06d}.json",
        )
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                os.remove(path)


def load(directory, step, device):
    step = _validate_step(step)
    _recover_transactions(directory)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    model = torch.load(os.path.join(directory, f"model_{step:06d}.pt"), map_location=device)
    optimizer = torch.load(os.path.join(directory, f"optimizer_{step:06d}.pt"), map_location=device)
    metadata = load_metadata(directory, step)
    return model, optimizer, metadata


def load_model(directory, step, device):
    step = _validate_step(step)
    _recover_transactions(directory)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = os.path.join(directory, f"model_{step:06d}.pt")
    return torch.load(path, map_location=device)


def load_metadata(directory, step):
    step = _validate_step(step)
    _recover_transactions(directory)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = os.path.join(directory, f"metadata_{step:06d}.json")
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("step") != step:
        raise ValueError(f"checkpoint metadata step does not match {step}")
    return metadata


def directory_identity(directory):
    """Return a stable identity for every file in a directory tree."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    files = sorted(candidate for candidate in directory.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"directory is empty: {directory}")
    digest = hashlib.sha256()
    entries = []
    for candidate in files:
        relative = candidate.relative_to(directory).as_posix()
        checksum = file_sha256(candidate)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\n")
        entries.append({"path": relative, "sha256": checksum, "bytes": candidate.stat().st_size})
    return {"path": str(directory), "sha256": digest.hexdigest(), "files": entries}


def checkpoint_identity(directory, step):
    """Return stable model and metadata identities for a completed checkpoint."""

    step = _validate_step(step)
    load_metadata(directory, step)
    directory = Path(directory).expanduser().resolve()
    model = directory / f"model_{step:06d}.pt"
    optimizer = directory / f"optimizer_{step:06d}.pt"
    metadata = directory / f"metadata_{step:06d}.json"
    return {
        "directory": str(directory),
        "step": step,
        "model_sha256": file_sha256(model),
        "optimizer_sha256": file_sha256(optimizer),
        "metadata_sha256": file_sha256(metadata),
    }


def load_timing(directory, step):
    """Load optional post-checkpoint timing, falling back to metadata for old checkpoints."""

    step = _validate_step(step)
    path = Path(directory) / f"timing_{step:06d}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
