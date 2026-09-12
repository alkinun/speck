"""Qualify CUDA document-NLL chunk parity for tokenizer-pilot evaluation."""

import json
import math
from pathlib import Path

import torch

from speck.checkpoint import load_model
from speck.io import file_sha256
from speck.tokenizer import Tokenizer
from speck.tokenizer_pilot_evaluation import (
    document_nll,
    iter_evaluation_documents,
    validate_evaluation_tokenizer,
)
from speck.tokenizer_pilot_runtime import load_pilot_run_manifest
from speck.tokenizer_pilot_train import build_pilot_model


def load_document_nll_policy(path):
    path = Path(path).resolve()
    policy = json.loads(path.read_text())
    if (
        policy.get("format") != "speck_tokenizer_pilot_document_nll_qualification_policy"
        or policy.get("format_version") != 1
        or policy.get("status") != "pre_output_cuda_chunk_parity_policy_frozen"
        or policy.get("sample", {}).get("documents") != 6
        or policy.get("requirements", {}).get("maximum_mean_nll_delta_nats_per_token") != 0.0001
        or policy.get("authority", {}).get("screen_execution") is not False
        or policy.get("authority", {}).get("D5_opening") is not False
    ):
        raise ValueError("unsupported tokenizer pilot document NLL policy")
    repository = path.parents[2]
    for identity in (
        *policy["inputs"].values(),
        *policy["qualified_implementation_inputs"].values(),
    ):
        candidate = Path(identity["path"])
        candidate = candidate if candidate.is_absolute() else repository / candidate
        if not candidate.is_file() or file_sha256(candidate) != identity["sha256"]:
            raise ValueError("tokenizer pilot document NLL policy identity mismatch")
    return {"path": str(path), "sha256": file_sha256(path), "value": policy}


def qualify_document_nll(policy, *, device="cuda"):
    """Compare one-chunk and production-chunk NLL on six preselected documents."""

    value = policy["value"]
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("tokenizer pilot document NLL qualification requires CUDA")
    run = load_pilot_run_manifest(value["inputs"]["run"]["path"])
    model = build_pilot_model(run, device)
    checkpoint_path = Path(value["inputs"]["checkpoint"]["path"])
    checkpoint_directory = checkpoint_path.parent
    step = int(checkpoint_path.stem.split("_")[-1])
    model.load_state_dict(load_model(checkpoint_directory, step, device))
    tokenizer = Tokenizer(run["tokenizer"]["model"]["path"])
    validate_evaluation_tokenizer(run, tokenizer)
    model.eval()
    limit = value["requirements"]["maximum_mean_nll_delta_nats_per_token"]
    results = []
    torch.cuda.reset_peak_memory_stats(device)
    for category in run["evaluation"]["categories"]:
        document = next(iter_evaluation_documents(category, verify_hash=True))
        token_ids = tokenizer.encode(document["text"], bos=True, eos=True)
        prediction_tokens = len(token_ids) - 1
        single = document_nll(
            model,
            token_ids,
            device=device,
            chunk_tokens=prediction_tokens,
        )
        chunked = document_nll(model, token_ids, device=device, chunk_tokens=4_096)
        mean_delta = abs(single - chunked) / prediction_tokens
        finite = all(math.isfinite(item) for item in (single, chunked, mean_delta))
        results.append(
            {
                "category": category["id"],
                "document_id": document["document_id"],
                "utf8_bytes": document["utf8_bytes"],
                "prediction_tokens": prediction_tokens,
                "single_chunk_nll_nats": single,
                "production_chunked_nll_nats": chunked,
                "mean_nll_delta_nats_per_token": mean_delta,
                "finite": finite,
                "within_tolerance": finite and mean_delta <= limit,
            }
        )
    passed = len(results) == value["requirements"]["categories_required"] and all(
        result["within_tolerance"] for result in results
    )
    if not passed:
        failures = [
            f"{result['category']}={result['mean_nll_delta_nats_per_token']:.9g}"
            for result in results
            if not result["within_tolerance"]
        ]
        raise RuntimeError(f"tokenizer pilot document NLL parity failed: {', '.join(failures)}")
    return {
        "format": "speck_tokenizer_pilot_document_nll_qualification",
        "format_version": 1,
        "status": "six_category_cuda_chunk_parity_pass_no_screen_authority",
        "policy": {"path": policy["path"], "sha256": policy["sha256"]},
        "run_id": run["run_id"],
        "run_fingerprint": run["run_fingerprint"],
        "checkpoint": value["inputs"]["checkpoint"],
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "results": results,
        "maximum_observed_mean_nll_delta_nats_per_token": max(
            result["mean_nll_delta_nats_per_token"] for result in results
        ),
        "maximum_allowed_mean_nll_delta_nats_per_token": limit,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "scientific_model_outputs_created": 0,
        "screen_execution_authority": False,
        "D5_opening_authority": False,
    }
