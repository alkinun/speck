"""Materialize and cross-runtime-check two permissive HELMET loader families."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from datasets.load import dataset_module_factory

EXPECTED_UUID = "b64b59d1-ea2c-4206-9171-b7cd739f3eff"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_identity(protocol):
    payload = {key: protocol[key] for key in protocol if key not in {"status", "result"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_rows(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
            + b"\n"
        )
    return digest.hexdigest()


def features_sha256(dataset):
    value = json.dumps(dataset.features.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def runtime_identity():
    return {
        "python": platform.python_version(),
        "datasets": importlib.metadata.version("datasets"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "dataset_module_factory_sha256": hashlib.sha256(
            inspect.getsource(dataset_module_factory).encode()
        ).hexdigest(),
    }


def validate_runtime(expected):
    observed = runtime_identity()
    for key, value in expected.items():
        if key in observed and observed[key] != value:
            raise ValueError(f"HELMET materializer runtime changed: {key}")
    return observed


def validate_volume(path, storage):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("HELMET materializer directory must be real and existing")
    mount = json.loads(
        subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "TARGET,FSTYPE,OPTIONS,UUID"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["filesystems"][0]
    options = set(mount["options"].split(","))
    if (
        mount["uuid"] != storage["device_uuid"]
        or mount["fstype"] != storage["filesystem"]
        or not set(storage["required_mount_options"]) <= options
        or path.stat().st_dev == Path("/").stat().st_dev
        or path.stat().st_uid != os.getuid()
        or oct(path.stat().st_mode & 0o777) != storage["required_directory_mode"]
    ):
        raise ValueError("HELMET materializer volume identity or permissions changed")
    return {
        "directory": str(path),
        "uuid": mount["uuid"],
        "filesystem": mount["fstype"],
        "options": sorted(options),
        "device_id": path.stat().st_dev,
        "mode": oct(path.stat().st_mode & 0o777),
    }


def _download(url, destination, expected):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "speck-evidence/1"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if temporary.stat().st_size != expected["bytes"] or file_sha256(temporary) != expected[
        "sha256"
    ]:
        raise ValueError("HELMET materializer direct payload changed")
    os.replace(temporary, destination)


def _verify_cached_raw_inputs(cache_dir, datasets):
    expected = {
        item["sha256"]: {**item, "dataset": dataset["id"]}
        for dataset in datasets
        for item in dataset["raw_inputs"]
    }
    found = {}
    for path in cache_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size not in {
            item["bytes"] for item in expected.values()
        }:
            continue
        digest = file_sha256(path)
        if digest in expected:
            found[digest] = str(path)
    if set(found) != set(expected):
        raise ValueError("HELMET legacy loader raw payloads do not match their pins")
    return [
        {**expected[digest], "cached_path": found[digest]}
        for digest in sorted(expected)
    ]


def _split_identity(dataset, expected):
    value = {
        "rows": dataset.num_rows,
        "canonical_rows_sha256": canonical_rows(dataset),
        "features_sha256": features_sha256(dataset),
    }
    if value != expected:
        raise ValueError(f"HELMET materializer split identity changed: {value}")
    return value


def _write_replayed_parquet(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = path.with_suffix(".first.parquet")
    second = path.with_suffix(".second.parquet")
    dataset.to_parquet(first)
    dataset.to_parquet(second)
    first_hash = file_sha256(first)
    if first_hash != file_sha256(second):
        raise ValueError("HELMET legacy Parquet replay is not byte-identical")
    os.replace(first, path)
    second.unlink()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": first_hash,
        "replay_byte_identical": True,
    }


def prepare(protocol, protocol_path, output_dir, cache_dir):
    root = Path(__file__).parents[1]
    environment = protocol["environments"]["legacy_materializer"]
    if file_sha256(root / environment["requirements"]) != environment["requirements_sha256"]:
        raise ValueError("HELMET materializer requirement lock changed")
    runtime = validate_runtime(environment)
    storage_root = Path(protocol["storage"]["root"])
    if output_dir.parent != storage_root or cache_dir.parent != storage_root:
        raise ValueError("HELMET materializer output/cache paths changed")
    storage = validate_volume(storage_root, protocol["storage"])
    if any(output_dir.iterdir()) or any(cache_dir.iterdir()):
        raise ValueError("HELMET materializer requires empty output and cache directories")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("HELMET materializer preflight must be unauthenticated")

    snapshots = {}
    loaded = {}
    for spec in protocol["datasets"]:
        loaded[spec["id"]] = load_dataset(
            spec["hub_repository"],
            revision=spec["hub_revision"],
            trust_remote_code=True,
            cache_dir=str(cache_dir / "datasets"),
        )
    raw_inputs = _verify_cached_raw_inputs(cache_dir, protocol["datasets"])

    banking = next(spec for spec in protocol["datasets"] if spec["id"] == "banking77")
    snapshots["banking77"] = {}
    for split, expected in banking["splits"].items():
        identity = _split_identity(loaded["banking77"][split], expected)
        artifact = _write_replayed_parquet(
            loaded["banking77"][split], output_dir / "banking77" / f"{split}.parquet"
        )
        snapshots["banking77"][split] = {**identity, **artifact}

    nlu = next(
        spec for spec in protocol["datasets"] if spec["id"] == "nlu_evaluation_data"
    )
    legacy_identity = _split_identity(loaded["nlu_evaluation_data"]["train"], nlu["splits"]["train"])
    nlu_path = output_dir / "nlu_evaluation_data" / "train.parquet"
    nlu_path.parent.mkdir(parents=True, exist_ok=True)
    _download(nlu["converted_parquet"]["url"], nlu_path, nlu["converted_parquet"])
    converted = load_dataset("parquet", data_files={"train": str(nlu_path)})["train"]
    converted_identity = _split_identity(converted, nlu["splits"]["train"])
    if converted_identity != legacy_identity:
        raise ValueError("NLU legacy and converted Parquet identities differ")
    snapshots["nlu_evaluation_data"] = {
        "train": {
            **converted_identity,
            "path": str(nlu_path),
            "bytes": nlu_path.stat().st_size,
            "sha256": file_sha256(nlu_path),
            "legacy_conversion_full_parity": True,
        }
    }
    return {
        "format": "speck_helmet_materializer_preflight",
        "format_version": 1,
        "status": "two_family_offline_materializer_strategy_qualified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "runtime": runtime,
        "storage": storage,
        "raw_inputs": raw_inputs,
        "snapshots": snapshots,
        "network": {
            "authentication": "none",
            "allowed_hosts": protocol["network"]["allowed_hosts"],
            "credentials_exposed": False,
        },
        "decision": {
            "strategy_qualified_for": ["banking77", "nlu_evaluation_data"],
            "helmet_execution_authorized": False,
            "extension_to_blocked_sources_authorized": False,
        },
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }


def check(protocol, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_runtime(protocol["environments"]["current_reader"])
    if (
        report.get("format") != "speck_helmet_materializer_preflight"
        or report.get("status") != "two_family_offline_materializer_strategy_qualified"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("decision", {}).get("helmet_execution_authorized") is not False
    ):
        raise ValueError("HELMET materializer preflight report is invalid")
    specs = {spec["id"]: spec for spec in protocol["datasets"]}
    for dataset_id, splits in report["snapshots"].items():
        for split, artifact in splits.items():
            path = Path(artifact["path"])
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or file_sha256(path) != artifact["sha256"]
            ):
                raise ValueError("HELMET materializer snapshot changed")
            loaded = load_dataset("parquet", data_files={split: str(path)})[split]
            identity = _split_identity(loaded, specs[dataset_id]["splits"][split])
            if any(artifact[key] != value for key, value in identity.items()):
                raise ValueError("HELMET current reader parity failed")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_materializer_preflight.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("HELMET materializer preflight runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_materializer_preflight_protocol":
        raise ValueError("HELMET materializer protocol has the wrong format")
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_qualified"}:
            raise ValueError("HELMET materializer protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve())
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("HELMET materializer protocol must be frozen and unexecuted")
        report = prepare(
            protocol,
            protocol_path,
            args.output_dir.expanduser().absolute(),
            args.cache_dir.expanduser().absolute(),
        )
        atomic_json(args.output, report)
    print(f"HELMET materializer preflight: {report['status']}")


if __name__ == "__main__":
    main()
