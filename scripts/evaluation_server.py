"""Launch the strict local OpenAI-compatible Speck evaluation server."""

import argparse

import torch

from speck.evaluation_server import TransformersEvaluationEngine, serve


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--allow-unattested-export", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    engine = TransformersEvaluationEngine.load(
        args.export,
        device=args.device,
        dtype=args.dtype,
        require_attestation=not args.allow_unattested_export,
    )
    serve(engine, args.host, args.port)


if __name__ == "__main__":
    main()
