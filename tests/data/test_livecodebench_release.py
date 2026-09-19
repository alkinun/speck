"""Release-level binding must reject incomplete files and mismatched loader evidence."""

import copy
import hashlib
import json

import pytest

from scripts.livecodebench_exclusion import prepare


def artifact(path, payload):
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def release(tmp_path):
    loader_bytes = b"ALLOWED_FILES = {'release_v6': ['test.jsonl', 'test2.jsonl']}\n"
    loader = artifact(tmp_path / "loader.py", loader_bytes)
    files, tree = [], []
    for i, name in enumerate(("test.jsonl", "test2.jsonl")):
        row = {
            "platform": "synthetic",
            "question_id": str(i),
            "contest_date": "2025-01-01",
            "question_title": "Example",
            "question_content": "Independent fixture.",
            "starter_code": "",
            "public_test_cases": "[]",
            "private_test_cases": "opaque",
        }
        f = artifact(tmp_path / name, (json.dumps(row) + "\n").encode())
        files.append({**f, "filename": name})
        tree.append({"type": "file", "path": name, "size": f["bytes"], "lfs": {"oid": f["sha256"]}})
    tree.append(
        {
            "type": "file",
            "path": "code_generation_lite.py",
            "oid": hashlib.sha1(
                b"blob " + str(len(loader_bytes)).encode() + b"\0" + loader_bytes
            ).hexdigest(),
        }
    )
    metadata = artifact(tmp_path / "tree.json", json.dumps(tree).encode())
    inputs = {
        "livecodebench": {
            "repo": "example/repo",
            "revision": "a" * 40,
            "release": "release_v6",
            "loader": loader,
            "metadata": metadata,
            "files": files,
        }
    }
    acquisition = {
        "repository": "example/repo",
        "revision": "a" * 40,
        "release": "release_v6",
        "files": copy.deepcopy(files),
    }
    return tmp_path, inputs, acquisition


def run(release):
    root, inputs, acquisition = release
    ip, ap = root / "inputs.json", root / "acquisition.json"
    ip.write_text(json.dumps(inputs))
    ap.write_text(json.dumps(acquisition))
    return prepare(ip, ap, root / "output")


def test_complete_release_is_projected(release):
    assert run(release)["rows"] == 2


@pytest.mark.parametrize("failure", ["missing", "duplicate", "revision", "size", "loader"])
def test_incomplete_or_inconsistent_release_fails(release, failure):
    root, inputs, acquired = release
    if failure == "missing":
        acquired["files"].pop()
    elif failure == "duplicate":
        acquired["files"].append(acquired["files"][0])
    elif failure == "revision":
        acquired["revision"] = "b" * 40
    elif failure == "size":
        acquired["files"][0]["bytes"] += 1
    else:
        inputs["livecodebench"]["loader"] = artifact(root / "changed.py", b"ALLOWED_FILES = {}\n")
    with pytest.raises(ValueError):
        run(release)
    assert not (root / "output").exists()
