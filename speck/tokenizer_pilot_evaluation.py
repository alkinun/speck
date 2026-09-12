"""Evaluate tokenizer-pilot checkpoints on paired whole documents."""

import hashlib
import json
from pathlib import Path

import torch

from speck.io import file_sha256
from speck.tokenizer_pilot_runtime import validate_pilot_run_manifest
from speck.validation import positive_integer

CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def validate_evaluation_tokenizer(run, tokenizer):
    expected = run["tokenizer"]
    if (
        tokenizer.fingerprint() != expected["model"]["sha256"]
        or tokenizer.vocab_size != expected["vocab_size"]
        or tokenizer.bos_id != expected["bos_token_id"]
        or tokenizer.eos_id != expected["eos_token_id"]
    ):
        raise ValueError("evaluation tokenizer differs from the pilot run")


def iter_evaluation_documents(category, *, verify_hash=True):
    """Yield identity-checked documents and verify declared category totals."""

    path = Path(category["path"])
    if not path.is_file() or (verify_hash and file_sha256(path) != category["sha256"]):
        raise ValueError(f"tokenizer pilot evaluation identity mismatch: {category['id']}")
    documents = 0
    utf8_bytes = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid tokenizer pilot evaluation JSON: {category['id']}:{line_number}"
                ) from error
            text = record.get("text")
            document_id = record.get("source_content_sha256")
            if (
                not isinstance(text, str)
                or not text
                or not isinstance(document_id, str)
                or hashlib.sha256(text.encode()).hexdigest() != document_id
                or record.get("category") != category["id"]
            ):
                raise ValueError(
                    f"invalid tokenizer pilot evaluation document: {category['id']}:{line_number}"
                )
            size = len(text.encode())
            documents += 1
            utf8_bytes += size
            yield {"document_id": document_id, "text": text, "utf8_bytes": size}
    if documents != category["documents"] or utf8_bytes != category["utf8_bytes"]:
        raise ValueError(f"tokenizer pilot evaluation totals mismatch: {category['id']}")


@torch.inference_mode()
def document_nll(model, token_ids, *, device, chunk_tokens=4_096):
    """Return exact next-token NLL while preserving state across bounded chunks."""

    chunk_tokens = positive_integer(chunk_tokens, "evaluation chunk tokens")
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        raise ValueError("evaluation documents must contain at least two tokens")
    length = len(token_ids) - 1
    if length > model.config.max_position_embeddings:
        raise ValueError("evaluation document exceeds the model context")
    device = torch.device(device)
    state = model.state(length=length, device=device)
    tokens = torch.tensor(token_ids, dtype=torch.int64, device=device)
    total = 0.0
    for start in range(0, length, chunk_tokens):
        end = min(start + chunk_tokens, length)
        inputs = tokens[start:end][None]
        targets = tokens[start + 1 : end + 1][None]
        loss = model(inputs, targets, state=state, loss_reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite tokenizer pilot document NLL")
        total += float(loss)
    if state.position != length:
        raise RuntimeError("evaluation state did not consume the complete document")
    return total


def evaluate_pilot_documents(
    run,
    model,
    tokenizer,
    *,
    device,
    chunk_tokens=4_096,
    verify_hashes=True,
):
    """Evaluate the six frozen categories with paired document-level output."""

    run = validate_pilot_run_manifest(run)
    validate_evaluation_tokenizer(run, tokenizer)
    categories = run["evaluation"]["categories"]
    if tuple(category.get("id") for category in categories) != CATEGORIES:
        raise ValueError("tokenizer pilot evaluation categories are incomplete or reordered")
    was_training = model.training
    model.eval()
    results = {}
    try:
        for category in categories:
            documents = []
            for document in iter_evaluation_documents(category, verify_hash=verify_hashes):
                token_ids = tokenizer.encode(document["text"], bos=True, eos=True)
                documents.append(
                    {
                        "document_id": document["document_id"],
                        "utf8_bytes": document["utf8_bytes"],
                        "nll_nats": document_nll(
                            model,
                            token_ids,
                            device=device,
                            chunk_tokens=chunk_tokens,
                        ),
                    }
                )
            results[category["id"]] = documents
    finally:
        model.train(was_training)
    return results
