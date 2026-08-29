"""Evaluate and summarize one SpeckGym v0 language checkpoint."""

import argparse
import json
import math
from pathlib import Path

from scripts import open_slm_eval
from scripts.base_checkpoint_export import export, validate_export
from speck.checkpoint import checkpoint_identity
from speck.common import base_dir
from speck.speckgym import load_speckgym_config
from speck.speckgym_eval import (
    cases_fingerprint,
    evaluate_procedural_checkpoint,
    generate_cases,
    resolve_language_checkpoint,
    training_metrics,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=tuple("ABCDE"))
    parser.add_argument("tokens", type=int)
    parser.add_argument(
        "stage", choices=("procedural", "standard", "summary", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "--experiment",
        default="experiments/SpeckGym-v0",
        help="SpeckGym experiment directory (default: %(default)s)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=float, help="standard-task smoke-test sample limit")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and (args.limit <= 0 or args.stage != "standard"):
        parser.error("--limit must be positive and is only supported by the standard stage")
    return args


def _output_dir(args):
    return (
        (
            args.output_dir
            or Path(base_dir()) / "evaluations" / "SpeckGym-v0" / args.run / str(args.tokens)
        )
        .expanduser()
        .resolve()
    )


def _standard(suite, run, tokens, output_dir, device, limit=None):
    _, checkpoint_dir, step, metadata = resolve_language_checkpoint(suite, run, tokens)
    identity = checkpoint_identity(checkpoint_dir, step)
    export_dir = output_dir / "model"
    identity_path = output_dir / "export_identity.json"
    if export_dir.exists():
        expected_export = {
            "checkpoint": identity,
            "local_model": open_slm_eval._local_model_identity(export_dir),
        }
        if (
            not identity_path.is_file()
            or json.loads(identity_path.read_text(encoding="utf-8")) != expected_export
        ):
            raise ValueError("local evaluation export belongs to a different checkpoint")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        export(checkpoint_dir, step, export_dir, metadata)
        validate_export(export_dir, metadata)
        expected_export = {
            "checkpoint": identity,
            "local_model": open_slm_eval._local_model_identity(export_dir),
        }
        identity_path.write_text(
            json.dumps(expected_export, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    config = open_slm_eval._load_config(Path(suite["evaluation"]["standard_config"]))
    result = open_slm_eval._run_lm_eval(
        config,
        output_dir / "standard",
        device,
        limit=limit,
        local_model=export_dir,
    )
    standard_identity = {
        "checkpoint": identity,
        "evaluation_config_sha256": open_slm_eval._sha256(
            Path(suite["evaluation"]["standard_config"])
        ),
        "limit": limit,
        "local_model": open_slm_eval._local_model_identity(export_dir),
        "result": {
            "path": result.relative_to(output_dir).as_posix(),
            "sha256": open_slm_eval._sha256(result),
        },
    }
    report_name = "standard_identity.json" if limit is None else "standard_smoke_identity.json"
    (output_dir / report_name).write_text(
        json.dumps(standard_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _standard_scores(suite, output_dir, checkpoint):
    identity_path = output_dir / "standard_identity.json"
    if not identity_path.is_file():
        return None
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("checkpoint") != checkpoint:
        raise ValueError("standard-task result belongs to a different checkpoint")
    config_path = Path(suite["evaluation"]["standard_config"])
    if identity.get("evaluation_config_sha256") != open_slm_eval._sha256(config_path):
        raise ValueError("standard-task evaluation config changed")
    model_identity = open_slm_eval._local_model_identity(output_dir / "model")
    if identity.get("local_model") != model_identity:
        raise ValueError("standard-task local model identity changed")
    path = output_dir / identity["result"]["path"]
    if not path.is_file() or open_slm_eval._sha256(path) != identity["result"]["sha256"]:
        raise ValueError("standard-task result identity changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["config"]["limit"] is not None:
        raise ValueError("cannot summarize a limited standard-task run")
    config = open_slm_eval._load_config(config_path)
    if result["lm_eval_version"] != config["lm_eval"]["version"]:
        raise ValueError("standard-task lm-eval version changed")
    if result["transformers_version"] != config["lm_eval"]["transformers_version"]:
        raise ValueError("standard-task Transformers version changed")
    scores = {}
    for task in config["lm_eval"]["tasks"]:
        samples = result["n-samples"][task]
        expected = config["lm_eval"]["expected_samples"][task]
        if samples["original"] != expected or samples["effective"] != expected:
            raise ValueError(f"standard-task result for {task} is incomplete")
        scores[task] = result["results"][task]["acc_norm,none"]
    return {"scores": scores, "result_sha256": open_slm_eval._sha256(path)}


def _procedural_scores(suite, procedural, run, tokens, actual_tokens, checkpoint):
    if procedural is None:
        return None
    evaluation = suite["evaluation"]
    cases = generate_cases(
        evaluation["seed"], evaluation["cases_per_family"], evaluation["families"]
    )
    expected_case_identity = {
        "seed": evaluation["seed"],
        "cases_per_family": evaluation["cases_per_family"],
        "families": evaluation["families"],
        "sha256": cases_fingerprint(cases),
    }
    if (
        procedural.get("run") != run
        or procedural.get("requested_tokens") != tokens
        or procedural.get("actual_tokens") != actual_tokens
        or procedural.get("checkpoint") != checkpoint
        or procedural.get("cases") != expected_case_identity
    ):
        raise ValueError("procedural result does not match the selected checkpoint and cases")
    return procedural["metrics"]


def _summary(suite, run, tokens, output_dir):
    _, checkpoint_dir, step, metadata = resolve_language_checkpoint(suite, run, tokens)
    checkpoint = checkpoint_identity(checkpoint_dir, step)
    procedural_path = output_dir / "procedural.json"
    procedural = (
        json.loads(procedural_path.read_text(encoding="utf-8"))
        if procedural_path.is_file()
        else None
    )
    report = {
        "format_version": 1,
        "run": run,
        "requested_tokens": tokens,
        "actual_tokens": metadata["global_tokens"],
        "checkpoint": checkpoint,
        "language_validation": {
            "loss": metadata["validation_loss"],
            "perplexity": min(1e9, math.exp(metadata["validation_loss"])),
            "evaluated_tokens": metadata["validation_tokens"],
            "global_tokens": metadata["validation_global_tokens"],
        },
        "procedural": _procedural_scores(
            suite, procedural, run, tokens, metadata["global_tokens"], checkpoint
        ),
        "standard": _standard_scores(suite, output_dir, checkpoint),
        "training": training_metrics(suite, run, tokens),
    }
    path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SpeckGym summary: {path}")
    return path


def main(argv=None):
    args = parse_args(argv)
    suite = load_speckgym_config(args.experiment)
    output_dir = _output_dir(args)
    if args.stage in {"procedural", "all"}:
        evaluate_procedural_checkpoint(
            suite,
            args.run,
            args.tokens,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
    if args.stage in {"standard", "all"}:
        _standard(suite, args.run, args.tokens, output_dir, args.device, args.limit)
    if args.stage in {"summary", "all"}:
        _summary(suite, args.run, args.tokens, output_dir)


if __name__ == "__main__":
    main()
