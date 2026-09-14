"""Freeze the pinned baseline artifact when the frozen screen projection exceeds budget."""

import argparse
import json
import shutil
from pathlib import Path

from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_artifact, repository_root
from speck.tokenization.pilot_screen import load_screen_report
from speck.tokenization.tokenizer import Tokenizer


def require_budget_stop(analysis):
    status = analysis.get("status")
    complete_excess = (
        status == "measured_budget_exceeded_retain_mistral_D5_unopened"
        and analysis.get("budget", {}).get("fits") is False
    )
    minimum_excess = (
        status == "projection_lower_bound_exceeded_retain_mistral_D5_unopened"
        and analysis.get("projection_lower_bound", {}).get("exceeds_ceiling") is True
    )
    if not (complete_excess or minimum_excess) or analysis.get("fallback") != "mistral-32k":
        raise ValueError("tokenizer fallback requires the frozen rule's decisive budget stop")


def freeze_budget_fallback(review_path, directory, output):
    review_path, directory, output = map(
        lambda path: Path(path).resolve(), (review_path, directory, output)
    )
    if directory.exists() or output.exists():
        raise FileExistsError("tokenizer freeze requires fresh artifact and decision paths")
    analysis = load_screen_report(review_path)
    require_budget_stop(analysis)
    review = json.loads(review_path.read_text())
    plan = json.loads((review_path.parent / review["plan"]["path"]).read_text())
    root = repository_root(__file__)
    static_path = repository_artifact(plan["static_prerequisite"]["path"], root)
    if file_sha256(static_path) != plan["static_prerequisite"]["sha256"]:
        raise ValueError("static baseline evidence identity mismatch")
    baseline = json.loads(static_path.read_text())["models"]["mistral-32k"]
    manifest_path = Path(baseline["manifest"]["path"])
    if file_sha256(manifest_path) != baseline["manifest"]["sha256"]:
        raise ValueError("baseline manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    model_path = manifest_path.parent / manifest["model"]["path"]
    if (
        manifest["id"] != "mistral-32k"
        or manifest["model"] != baseline["model"]
        or file_sha256(model_path) != manifest["model"]["sha256"]
    ):
        raise ValueError("baseline model differs from the frozen static prerequisite")
    tokenizer = Tokenizer(model_path)
    if (tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id) != (
        32000,
        1,
        2,
        0,
    ):
        raise ValueError("baseline tokenizer geometry differs from the declared fallback")
    directory.mkdir(parents=True)
    shutil.copyfile(model_path, directory / "tokenizer.model")
    metadata = {
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "filename": "tokenizer.model",
        "vocab_size": tokenizer.vocab_size,
        "fingerprint": tokenizer.fingerprint(),
    }
    durable_json(directory / "tokenizer_metadata.json", metadata)
    Tokenizer.load(directory, repo=manifest["repo"], revision=manifest["revision"])
    decision = {
        "format": "speck_tokenizer_decision",
        "format_version": 1,
        "status": "tokenizer_selected_and_frozen",
        "selected_tokenizer": "mistral-32k",
        "basis": "v10 measured-confirmation-budget stop; baseline fallback, not a replicated custom-quality conclusion",
        "screen_analysis": analysis,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "tokenizer": {
            "directory": str(directory),
            "repo": manifest["repo"],
            "revision": manifest["revision"],
        },
        "artifacts": [
            {"path": str(directory / name), "sha256": file_sha256(directory / name)}
            for name in ("tokenizer.model", "tokenizer_metadata.json")
        ],
        "baseline_manifest": baseline["manifest"],
        "base_vocab_size": 32000,
        "reserved_role_capacity": {
            "additional_ids": 3,
            "effective_model_vocab_size": 32003,
            "scope": "Existing model-accounting reservation only; no post-training recipe decision",
        },
        "all_attempt_spending_complete": analysis["projection_lower_bound"][
            "all_attempt_spending_complete"
        ],
        "D5_opened_by_this_decision": False,
        "confirmation_launched": False,
        "training_authority": False,
        "implementation": [
            {"path": str(root / name), "sha256": file_sha256(root / name)}
            for name in (
                "scripts/tokenizer_budget_fallback.py",
                "speck/tokenization/pilot_screen.py",
                "speck/tokenization/pilot.py",
                "speck/tokenization/tokenizer.py",
            )
        ],
        "remaining": "Reconcile historical all-attempt spending for disclosure. Bind this tokenizer to per-arm source views, packing and model/launch manifests; the budget fallback does not claim custom inferiority or authorize model training.",
    }
    durable_json(output, decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = freeze_budget_fallback(args.review, args.directory, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "selected_tokenizer",
                    "tokenizer_fingerprint",
                    "all_attempt_spending_complete",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
