"""Build a public-text exclusion projection from a pinned local LiveCodeBench release."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from speck.data.sources.livecodebench_exclusion import project_files
from speck.provenance.io import atomic_json, file_sha256


def verify_artifact(artifact):
    path = Path(artifact["path"])
    if file_sha256(path) != artifact["sha256"]:
        raise ValueError("input evidence checksum mismatch")
    return path


def prepare(inputs_path, acquisition_path, output_dir):
    inputs = json.loads(Path(inputs_path).read_text())
    acquisition = json.loads(Path(acquisition_path).read_text())
    pinned = inputs["livecodebench"]
    if (acquisition["repository"], acquisition["revision"], acquisition["release"]) != (
        pinned["repo"],
        pinned["revision"],
        pinned["release"],
    ):
        raise ValueError("acquisition does not match pinned release")
    tree = json.loads(verify_artifact(pinned["metadata"]).read_text())
    listed = {e["path"]: e for e in tree if e["type"] == "file"}
    expected = {e["filename"]: e for e in pinned["files"]}
    if len(expected) != len(pinned["files"]):
        raise ValueError("duplicate pinned filename")
    loader = verify_artifact(pinned["loader"])
    loader_bytes = loader.read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(loader_bytes)).encode() + b"\0" + loader_bytes
    ).hexdigest()
    if blob != listed["code_generation_lite.py"]["oid"]:
        raise ValueError("loader differs from pinned upstream tree")
    # Only inspect literal data in the pinned loader; never import or execute it.
    release_map = next(
        ast.literal_eval(node.value)
        for node in ast.parse(loader.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ALLOWED_FILES" for t in node.targets)
    )
    order = release_map[pinned["release"]]
    if len(order) != len(set(order)) or set(order) != set(expected):
        raise ValueError("pinned file set differs from upstream release definition")
    actual = {e["filename"]: e for e in acquisition["files"]}
    if len(actual) != len(acquisition["files"]) or set(actual) != set(expected):
        raise ValueError("incomplete or duplicate acquisition")
    for name, entry in expected.items():
        identities = [
            (entry["bytes"], entry["sha256"]),
            (actual[name]["bytes"], actual[name]["sha256"]),
            (listed[name]["size"], listed[name]["lfs"]["oid"]),
        ]
        if len(set(identities)) != 1:
            raise ValueError("acquisition, pin and upstream tree disagree")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("choose a fresh output directory")
    output_dir.mkdir(parents=True)
    result = project_files([actual[name] for name in order], output_dir / "public-text.jsonl")
    result.update(
        repository=pinned["repo"],
        revision=pinned["revision"],
        release=pinned["release"],
        input_manifest_sha256=file_sha256(inputs_path),
        acquisition_sha256=file_sha256(acquisition_path),
        loader_sha256=file_sha256(loader),
        boundary="Entire pinned release for exclusion only. Private tests, canonical solutions "
        "and future releases are not covered by this public-text projection. No scoring protocol.",
    )
    atomic_json(output_dir / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = prepare(args.inputs, args.acquisition, args.output_dir)
    print(json.dumps({k: result[k] for k in ("rows", "platform_counts", "date_min", "date_max")}))


if __name__ == "__main__":
    main()
