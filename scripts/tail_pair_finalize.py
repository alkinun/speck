"""Finalize matched tail checkpoints and checkpoint averages for evaluation."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from scripts.checkpoint_average import average_checkpoints, average_identity, write_average
from speck.checkpoint import checkpoint_identity, latest, load_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("--control-checkpoints", type=Path, required=True)
    parser.add_argument("--constant-checkpoints", type=Path, required=True)
    parser.add_argument("--average-steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pair(pair_dir):
    path = Path(pair_dir).expanduser().resolve() / "pair.json"
    if not path.is_file():
        raise FileNotFoundError(f"tail pair manifest does not exist: {path}")
    pair = json.loads(path.read_text(encoding="utf-8"))
    if pair.get("format") != "speck_tail_pair" or pair.get("format_version") != 1:
        raise ValueError("unsupported tail pair format")
    return path, pair


def validate_endpoint(pair, arm, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    step = latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no completed {arm} checkpoint in {checkpoint_dir}")
    metadata = load_metadata(checkpoint_dir, step)
    resolved = metadata["resolved"]
    expected = pair[arm]
    if (
        step != resolved["steps"]
        or resolved.get("parent_checkpoint") != pair["parent_checkpoint"]
        or resolved.get("branch_schedule") != expected["schedule"]
        or resolved.get("run") != expected["run"]
        or resolved.get("train_tokens") != pair["train_tokens"]
        or resolved.get("consumed_tokens") != pair["consumed_tokens"]
        or resolved.get("world_size") != pair["world_size"]
        or resolved.get("global_token_offset") != pair["parent_global_tokens"]
        or metadata.get("manifest") != pair["manifest"]
        or metadata.get("global_tokens") != pair["parent_global_tokens"] + pair["consumed_tokens"]
    ):
        raise ValueError(f"{arm} checkpoint does not match the tail pair")
    return metadata, checkpoint_identity(checkpoint_dir, step)


def finalize(pair_dir, control_dir, constant_dir, average_steps, output_dir):
    pair_path, pair = load_pair(pair_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"tail pair finalization already exists: {output_dir}")
    if len(average_steps) < 2 or average_steps != sorted(set(average_steps)):
        raise ValueError("average steps must contain at least two sorted unique steps")

    control_metadata, control_final = validate_endpoint(pair, "control", control_dir)
    constant_metadata, constant_final = validate_endpoint(pair, "constant", constant_dir)
    if control_metadata["global_tokens"] != constant_metadata["global_tokens"]:
        raise ValueError("tail pair final checkpoints have different global token positions")
    if average_steps[-1] > min(control_metadata["step"], constant_metadata["step"]):
        raise ValueError("average window extends beyond a final checkpoint")

    building = output_dir.with_name(output_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        control_state, control_average = average_checkpoints(control_dir, average_steps)
        write_average(building / "control-average", control_state, control_average)
        del control_state
        control_average_identity = average_identity(building / "control-average")
        constant_state, constant_average = average_checkpoints(constant_dir, average_steps)
        write_average(building / "constant-average", constant_state, constant_average)
        del constant_state
        constant_average_identity = average_identity(building / "constant-average")
        report = {
            "format": "speck_tail_pair_finalization",
            "format_version": 1,
            "pair_manifest": {
                "path": str(pair_path),
                "sha256": _sha256(pair_path),
            },
            "global_tokens": control_metadata["global_tokens"],
            "average_steps": average_steps,
            "control": {
                "final_checkpoint": control_final,
                "average": {
                    "path": "control-average",
                    "model_sha256": control_average_identity["model_sha256"],
                    "metadata_sha256": control_average_identity["metadata_sha256"],
                },
            },
            "constant": {
                "final_checkpoint": constant_final,
                "average": {
                    "path": "constant-average",
                    "model_sha256": constant_average_identity["model_sha256"],
                    "metadata_sha256": constant_average_identity["metadata_sha256"],
                },
            },
        }
        (building / "finalization.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return report


def main():
    args = parse_args()
    report = finalize(
        args.pair_dir,
        args.control_checkpoints,
        args.constant_checkpoints,
        args.average_steps,
        args.output_dir,
    )
    print(
        f"Finalized matched tail pair at {report['global_tokens']:,} global tokens "
        f"with {len(report['average_steps'])} averaged checkpoints"
    )


if __name__ == "__main__":
    main()
