"""Qualify an attested Speck export against external-suite endpoint request shapes."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from speck.evaluation_server import TransformersEvaluationEngine, exercise_endpoint

REQUIRED_EXPORT_FILES = (
    "architecture_speck.py",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "modeling_speck.py",
    "native_speck.py",
    "padding_speck.py",
    "speck_parity.json",
    "speck_source.json",
    "tokenization_speck.py",
    "tokenizer.model",
    "tokenizer_config.json",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision():
    root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("endpoint qualification requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def qualify(export_dir, *, device, dtype, runner_revision):
    export_dir = Path(export_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_EXPORT_FILES if not (export_dir / name).is_file()]
    if missing:
        raise ValueError(f"instruction export is missing qualification files: {missing}")
    parity = json.loads((export_dir / "speck_parity.json").read_text(encoding="utf-8"))
    source = json.loads((export_dir / "speck_source.json").read_text(encoding="utf-8"))
    engine = TransformersEvaluationEngine.load(export_dir, device=device, dtype=dtype)
    endpoint = exercise_endpoint(engine)
    files = {
        name: {
            "bytes": (export_dir / name).stat().st_size,
            "sha256": file_sha256(export_dir / name),
        }
        for name in REQUIRED_EXPORT_FILES
    }
    return {
        "format": "speck_evaluation_endpoint_qualification",
        "format_version": 1,
        "status": "qualified_for_serialized_openai_correctness_evaluation",
        "created_on": datetime.now(timezone.utc).date().isoformat(),
        "runner_revision": runner_revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(engine.device),
            "dtype": dtype,
        },
        "export": {
            "model_id": engine.model_id,
            "maximum_context": engine.maximum_context,
            "source": source,
            "parity": parity,
            "files": files,
        },
        "endpoint": endpoint,
        "qualified_consumers": {
            "nolima": "pinned AsyncOpenAI chat request shape",
            "ruler": "pinned NeMo-Skills OpenAI chat shape with presence_penalty=0",
        },
        "limitations": [
            "serialized correctness adapter; no serving-throughput claim",
            "short request-shape smoke; no benchmark task or long-context capability claim",
            "each evaluated candidate still requires its own attested export and context ceiling",
        ],
    }


def main(argv=None):
    args = arguments(argv)
    report = qualify(
        args.export,
        device=args.device,
        dtype=args.dtype,
        runner_revision=repository_revision(),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
