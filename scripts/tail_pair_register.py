"""Register benchmark evidence for one finalized tail-pair model variant."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from speck.checkpoint import directory_identity

_VARIANTS = {
    "control-final": ("control", "final_checkpoint", "checkpoint"),
    "constant-final": ("constant", "final_checkpoint", "checkpoint"),
    "control-average": ("control", "average", "average"),
    "constant-average": ("constant", "average", "average"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finalization_dir", type=Path)
    parser.add_argument("variant", choices=tuple(_VARIANTS))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--open-slm", type=Path, default=None)
    parser.add_argument("--bananamind", type=Path, default=None)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, value


def _validate_source(finalization, variant, source):
    arm, artifact, source_type = _VARIANTS[variant]
    if (
        source.get("format") != "speck_export_source"
        or source.get("format_version") != 1
        or source.get("type") != source_type
    ):
        raise ValueError("export source type does not match the result variant")
    expected = finalization[arm][artifact]
    if source_type == "checkpoint":
        if source.get("checkpoint") != expected:
            raise ValueError("export checkpoint does not match finalization")
    else:
        actual = source.get("average", {})
        if any(actual.get(key) != expected[key] for key in ("model_sha256", "metadata_sha256")):
            raise ValueError("export average does not match finalization")


def _open_slm_result(path, model_identity):
    path, result = _load_json(path, "Open-SLM summary")
    scores = result.get("scores_percent")
    if not isinstance(scores, dict):
        raise ValueError("Open-SLM summary has no scores_percent")
    recorded_model = result.get("model", {}).get("transformers_export")
    if recorded_model is not None and recorded_model != model_identity:
        raise ValueError("Open-SLM summary records a different model export")
    return {"path": str(path), "sha256": _sha256(path), "scores_percent": scores}


def _bananamind_result(path, model_dir):
    path, result = _load_json(path, "BananaMind report")
    summary = result.get("summary")
    if not isinstance(summary, dict) or summary.get("official_complete_run") is not True:
        raise ValueError("BananaMind report is not an official complete run")
    model = result.get("model")
    if not isinstance(model, str) or Path(model).expanduser().resolve() != model_dir:
        raise ValueError("BananaMind report records a different model export")
    return {"path": str(path), "sha256": _sha256(path), "summary": summary}


def register(finalization_dir, variant, model_dir, open_slm=None, bananamind=None):
    if open_slm is None and bananamind is None:
        raise ValueError("provide at least one benchmark result")
    finalization_dir = Path(finalization_dir).expanduser().resolve()
    finalization_path, finalization = _load_json(
        finalization_dir / "finalization.json", "tail-pair finalization"
    )
    if (
        finalization.get("format") != "speck_tail_pair_finalization"
        or finalization.get("format_version") != 1
    ):
        raise ValueError("unsupported tail-pair finalization format")
    model_dir = Path(model_dir).expanduser().resolve()
    source_path, source = _load_json(model_dir / "speck_source.json", "export source")
    _validate_source(finalization, variant, source)
    model_identity = directory_identity(model_dir)

    record = {
        "format": "speck_tail_pair_result",
        "format_version": 1,
        "variant": variant,
        "finalization": {
            "path": str(finalization_path),
            "sha256": _sha256(finalization_path),
        },
        "model_export": model_identity,
        "source_sha256": _sha256(source_path),
    }
    if open_slm is not None:
        record["open_slm"] = _open_slm_result(open_slm, model_identity)
    if bananamind is not None:
        record["bananamind"] = _bananamind_result(bananamind, model_dir)

    results_dir = finalization_dir / "results"
    results_dir.mkdir(exist_ok=True)
    output = results_dir / f"{variant}.json"
    if output.exists():
        raise FileExistsError(f"tail-pair result already exists: {output}")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return record


def main():
    args = parse_args()
    register(
        args.finalization_dir,
        args.variant,
        args.model_dir,
        args.open_slm,
        args.bananamind,
    )
    print(f"Registered {args.variant} benchmark evidence")


if __name__ == "__main__":
    main()
