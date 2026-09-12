"""Qualify corrected tokenizer-pilot manifests without creating model outputs."""

import json
from pathlib import Path

import torch

from speck.io import file_sha256
from speck.tokenizer_pilot_evaluation import iter_evaluation_documents
from speck.tokenizer_pilot_runtime import (
    PilotBatchLoader,
    PilotTokenStream,
    load_pilot_run_manifest,
)


def _verified_hash(path, expected, cache):
    path = Path(path)
    actual = cache.setdefault(str(path), file_sha256(path))
    if actual != expected:
        raise ValueError(f"tokenizer pilot qualification identity mismatch: {path}")
    return actual


def _loader_probe(stream, offset):
    settings = stream.run["settings"]
    stride = settings["sequence_length"] * settings["device_batch_size"]
    state = {
        "format_version": 1,
        "contract": "tokenizer_pilot_batch_start",
        "run_fingerprint": stream.run["run_fingerprint"],
        "token_offset": offset,
        "sequence_length": settings["sequence_length"],
        "batch_size": settings["device_batch_size"],
    }
    loader = PilotBatchLoader(stream, resume_state=state)
    inputs, targets, returned = next(loader)
    if returned != state or not torch.equal(inputs[:, 1:], targets[:, :-1]):
        raise ValueError("tokenizer pilot loader probe broke causal token continuity")
    if loader.state_dict()["token_offset"] != offset + stride:
        raise ValueError("tokenizer pilot loader probe advanced by the wrong stride")
    return {
        "offset": offset,
        "next_offset": loader.state_dict()["token_offset"],
        "first_input": int(inputs[0, 0]),
        "first_target": int(targets[0, 0]),
    }


def qualify_corrected_runtime(materialization_path):
    """Verify all v2 bytes, evaluation rows, boundaries, and resume probes."""

    materialization_path = Path(materialization_path).resolve()
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("format")
        != "speck_tokenizer_pilot_corrected_screen_materialization_result"
        or materialization.get("format_version") != 2
        or materialization.get("status")
        != "three_corrected_screen_runs_materialized_execution_blocked"
        or materialization.get("authority", {}).get("screen_execution") is not False
    ):
        raise ValueError("unsupported corrected tokenizer pilot materialization")
    declarations = materialization.get("runs")
    if not isinstance(declarations, list) or len(declarations) != 3:
        raise ValueError("corrected tokenizer pilot materialization must contain three runs")
    cache = {}
    evaluation_cache = {}
    run_results = []
    for declaration in declarations:
        run_path = materialization_path.parent / declaration["path"]
        run = load_pilot_run_manifest(run_path)
        if (
            run["run_id"] != declaration["run_id"]
            or run["run_fingerprint"] != declaration["run_fingerprint"]
        ):
            raise ValueError("materialized tokenizer pilot run identity mismatch")
        _verified_hash(
            run["tokenizer"]["model"]["path"], run["tokenizer"]["model"]["sha256"], cache
        )
        for identity in (
            run["training_data"]["fixed_stream"],
            run["training_data"]["continuation"],
            run["evaluation"]["sample"],
            run["flop_correction"],
        ):
            _verified_hash(identity["path"], identity["sha256"], cache)
        stream = PilotTokenStream(run, verify_hashes=True)
        for shard in stream.shard_manifests:
            cache[str(Path(shard["path"]))] = shard["sha256"]
        stride = run["settings"]["sequence_length"] * run["settings"]["device_batch_size"]
        final_offset = run["stops"]["run_stop_aligned_tokens"] - stride
        probes = [_loader_probe(stream, 0), _loader_probe(stream, final_offset)]
        resume_offset = run["settings"]["batch_tokens"]
        if resume_offset < final_offset:
            probes.insert(1, _loader_probe(stream, resume_offset))
        evaluation = []
        for category in run["evaluation"]["categories"]:
            key = (category["path"], category["sha256"])
            if key not in evaluation_cache:
                documents = list(iter_evaluation_documents(category, verify_hash=True))
                evaluation_cache[key] = {
                    "documents": len(documents),
                    "utf8_bytes": sum(document["utf8_bytes"] for document in documents),
                }
                cache[category["path"]] = category["sha256"]
            evaluation.append({"id": category["id"], **evaluation_cache[key]})
        run_results.append(
            {
                "run_id": run["run_id"],
                "run_fingerprint": run["run_fingerprint"],
                "tokenizer_id": run["tokenizer_id"],
                "fixed_tokens": stream.fixed_tokens,
                "available_tokens": stream.total_tokens,
                "run_stop_aligned_tokens": stream.maximum_offset,
                "shards": len(stream.shards),
                "resume_probes": probes,
                "evaluation": evaluation,
            }
        )
    return {
        "format": "speck_tokenizer_pilot_runtime_qualification",
        "format_version": 1,
        "status": "corrected_streams_and_evaluation_inputs_qualified_no_model_outputs",
        "materialization": {
            "path": str(materialization_path),
            "sha256": file_sha256(materialization_path),
            "plan_fingerprint": materialization["plan_fingerprint"],
        },
        "runs": run_results,
        "unique_files_verified": len(cache),
        "unique_evaluation_files_parsed": len(evaluation_cache),
        "gates": {
            "run_fingerprints": "pass_all",
            "tokenizer_and_parent_manifests": "pass_all",
            "packed_shard_sizes_and_sha256": "pass_all",
            "fixed_and_continuation_boundaries": "pass_all",
            "initial_resume_and_final_batch_cursors": "pass_all",
            "evaluation_sha256_document_identity_and_totals": "pass_all",
            "model_outputs_created": 0,
        },
        "authority": {
            "runtime_inputs_qualified": True,
            "screen_execution": False,
            "confirmation_execution": False,
            "D5_opening": False,
            "flagship_training": False,
        },
    }
