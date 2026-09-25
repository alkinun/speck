"""Exercise actual packed shards and fresh-process cursor replay without model training."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from speck.config import load_experiment
from speck.data.dataset import load_manifest, resolve_data_dir, verify_shards
from speck.data.loader import manifest_fingerprint, packed_loader, sequence_schedule
from speck.model import build_model
from speck.operations.runtime import dist_info
from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.tokenizer import get_tokenizer


def _batch_identity(inputs, targets, state):
    return {
        "inputs_sha256": hashlib.sha256(inputs.contiguous().numpy().tobytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(targets.contiguous().numpy().tobytes()).hexdigest(),
        "state": state,
    }


def check_loader(experiment, output_dir, *, batches=64, mode="scan"):
    """Scan a finite schedule or compare saved batches in a new Python process."""

    if mode not in {"scan", "replay"} or type(batches) is not int or batches < 2:
        raise ValueError("loader check requires scan/replay mode and at least two batches")
    torch.set_num_threads(1)
    configs = load_experiment(experiment, "model", "data", "train", "tokenizer")
    data, train = configs["data"], configs["train"]
    directory = resolve_data_dir(data.get("output_dir"), data.get("output_name"))
    manifest = load_manifest(directory)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    rank, _, world_size = dist_info()
    stride = train["device_batch_size"] * train["sequence_length"] * world_size
    if batches * stride > manifest["requested_train_tokens"]:
        raise ValueError("loader check would exceed the finite training schedule")
    if rank == 0:
        verify_shards(directory, manifest)
    with torch.device("meta"):
        model = build_model(
            configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
        )
        parameters = model.parameter_count()
    contract = {
        "manifest": manifest_fingerprint(manifest),
        "tokenizer": tokenizer.fingerprint(),
        "sequence_length": train["sequence_length"],
        "batch_size": train["device_batch_size"],
        "world_size": world_size,
        "rank": rank,
        "batches": batches,
        "loader_sha256": file_sha256(Path(__file__).with_name("loader.py")),
    }
    output_dir = Path(output_dir)
    scan_path = output_dir / f"rank-{rank}.scan.json"
    result_path = output_dir / f"rank-{rank}.{mode}.json"
    if result_path.exists():
        raise FileExistsError(f"loader check output already exists: {result_path}")
    options = {
        "batch_size": train["device_batch_size"],
        "sequence_length": train["sequence_length"],
        "split": "train",
        "device": "cpu",
        "data_dir": directory,
    }
    if mode == "replay":
        original = json.loads(scan_path.read_text())
        if original["contract"] != contract:
            raise ValueError("loader replay contract changed")
        expected = original["replay_batches"]
        loader = packed_loader(tokenizer, resume_state_dict=expected[0]["state"], **options)
        try:
            for batch in expected:
                if _batch_identity(*next(loader)) != batch:
                    raise AssertionError("fresh-process loader replay differs from the original")
        finally:
            loader.close()
        result = {"contract": contract, "status": "exact_replay", "replayed_batches": len(expected)}
    else:
        loader = packed_loader(tokenizer, **options)
        counts, replay = Counter(), []
        start = batches // 2
        try:
            for index in range(batches):
                inputs, targets, state = next(loader)
                if state["global_consumed_tokens"] != index * stride:
                    raise AssertionError("loader cursor does not match the consumed-token boundary")
                if any(state["source_epochs"].values()):
                    raise AssertionError("finite pilot unexpectedly repeated a source")
                if (
                    min(inputs.min().item(), targets.min().item()) < 0
                    or max(inputs.max().item(), targets.max().item()) >= tokenizer.vocab_size
                ):
                    raise AssertionError("packed token is outside the frozen tokenizer vocabulary")
                if manifest["mixture"].get("schedule") is None:
                    counts[state["selected_source"]] += inputs.numel()
                else:
                    rows = train["device_batch_size"]
                    first = index * stride // train["sequence_length"] + rank * rows
                    for source_id, _ in sequence_schedule(manifest, first, rows):
                        counts[source_id] += train["sequence_length"]
                if start <= index < start + 8:
                    replay.append(_batch_identity(inputs, targets, state))
        finally:
            loader.close()
        result = {
            "contract": contract,
            "status": "finite_scan_passed",
            "parameters": parameters,
            "rank_training_tokens": sum(counts.values()),
            "rank_tokens_by_source": dict(counts),
            "replay_batches": replay,
            "boundary": "CPU packed-loader checks only; no CUDA transfers, collectives, or model training.",
        }
    atomic_json(result_path, result)
    return result
