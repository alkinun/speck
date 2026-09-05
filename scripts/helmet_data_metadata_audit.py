"""Audit HELMET dataset metadata, active configs, licenses, and storage without payload download."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from speck.external import qualify_external_suite, validate_external_suite

DATA_REVISION = "dddb209d03e38f1f0faf76d6d05ef4ccf96240ee"
README_SHA256 = "07e0d14a47da6b583061376da95a3f6d0b71f3ee25026438439a93372e56f6fd"
DOWNLOAD_SCRIPT_SHA256 = "dcf5eb0c71dc2a78886e4e4fe09dfa0784722d836a24f2464c48160381aade5e"
ARCHIVES = {
    "data.tar.gz": {
        "git_object_id": "f4c45800e388be17ace3846c3ca10e8e282aadc5",
        "pointer_bytes": 136,
        "oid_sha256": "9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9",
        "payload_bytes": 11_271_916_108,
        "role": "required by pinned download script and benchmark configs",
    },
    "data_v2.tar.gz": {
        "git_object_id": "f1bdffc0816dec7966e6f5a267c7f22d79ab0901",
        "pointer_bytes": 135,
        "oid_sha256": "89e6f3a197c6079d6b3b9092d9cf4503e4989bdc0bd659741bbb4e43bfa7600e",
        "payload_bytes": 8_858_479_649,
        "role": "excluded; added after pinned code path and not referenced by its configs",
    },
}
CATEGORIES = ("recall", "rag", "rerank", "icl", "longqa", "summ", "cite")
CONFIGS = tuple(
    f"configs/{category}{suffix}.yaml"
    for category in CATEGORIES
    for suffix in ("_short", "")
)
DATASET_FAMILIES = {
    "recall": ["RULER", "Lost-in-the-Middle JSON KV"],
    "rag": ["KILT", "Natural Questions", "TriviaQA", "HotpotQA", "PopQA"],
    "rerank": ["MS MARCO passage ranking"],
    "icl": ["TREC", "Banking77", "CLINC150", "NLU evaluation data"],
    "longqa": ["NarrativeQA", "InfiniteBench"],
    "summ": ["InfiniteBench", "Multi-LexSum"],
    "cite": ["ALCE", "ASQA", "QAMPARI"],
}
MINIMUM_VOLUME_FREE_BYTES = 64 * 1024**3


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--code-checkout", type=Path, required=True)
    parser.add_argument("--data-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-device", default="/dev/sda2")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_revision():
    return _git(Path(__file__).parents[1], "rev-parse", "HEAD")


def _git(directory, *args, binary=False):
    env = os.environ.copy()
    env.update(
        {
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "remote.origin.promisor",
            "GIT_CONFIG_VALUE_0": "false",
        }
    )
    result = subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=not binary,
        env=env,
    )
    return result.stdout if binary else result.stdout.strip()


def _tree(checkout):
    entries = {}
    for line in _git(checkout, "ls-tree", "-r", "-l", DATA_REVISION).splitlines():
        match = re.fullmatch(r"(\d+) (\w+) ([0-9a-f]{40}) +(-|\d+)\t(.+)", line)
        if match is None:
            raise ValueError(f"cannot parse HELMET data tree entry: {line}")
        mode, kind, object_id, size, path = match.groups()
        entries[path] = {
            "path": path,
            "mode": mode,
            "type": kind,
            "git_object_id": object_id,
            "bytes": None if size == "-" else int(size),
        }
    return entries


def _show(checkout, path):
    return _git(checkout, "show", f"{DATA_REVISION}:{path}", binary=True)


def _lfs_pointer(value):
    match = re.fullmatch(
        rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n",
        value,
    )
    if match is None:
        raise ValueError("HELMET archive is not a canonical Git-LFS pointer")
    return {"oid_sha256": match.group(1).decode(), "payload_bytes": int(match.group(2))}


def _pack_snapshot(checkout):
    git_dir = Path(checkout) if (Path(checkout) / "objects").is_dir() else Path(checkout) / ".git"
    entries = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted((git_dir / "objects/pack").iterdir())
        if path.is_file()
    ]
    return {"files": entries, "sha256": bytes_sha256(json.dumps(entries, sort_keys=True).encode())}


def _parse_config(path, category, short):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = str(value["datasets"]).split(",")
    input_lengths = [int(item) for item in str(value["input_max_length"]).split(",")]
    generation_lengths = [
        int(item) for item in str(value["generation_max_length"]).split(",")
    ]
    expected = {8192, 16384, 32768, 65536} if short else {131072}
    if (
        len(datasets) != len(input_lengths)
        or len(datasets) != len(generation_lengths)
        or set(input_lengths) != expected
    ):
        raise ValueError(f"HELMET config geometry changed: {path.name}")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "category": category,
        "short_config": short,
        "entries": len(datasets),
        "input_lengths": input_lengths,
        "generation_lengths": generation_lengths,
        "datasets": datasets,
        "max_test_samples": value["max_test_samples"],
        "use_chat_template": value["use_chat_template"],
    }


def _device(path):
    result = subprocess.run(
        [
            "lsblk",
            "-b",
            "-J",
            "-o",
            "NAME,PATH,TYPE,FSTYPE,SIZE,FSAVAIL,MOUNTPOINTS,UUID,MODEL",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    device = json.loads(result.stdout)["blockdevices"][0]
    return {
        "path": device["path"],
        "type": device["type"],
        "filesystem": device["fstype"],
        "bytes": device["size"],
        "available_bytes": device["fsavail"],
        "mountpoints": device["mountpoints"],
        "uuid": device["uuid"],
        "model": device["model"],
    }


def _stable_audit(contract_path, code, data):
    contract = validate_external_suite(contract_path)
    if contract["suite_id"] != "helmet" or contract["data"]["revision"] != DATA_REVISION:
        raise ValueError("HELMET metadata audit received a different contract")
    source = qualify_external_suite(contract_path, {contract["upstream"]["id"]: code})
    before = _pack_snapshot(data)
    tree = _tree(data)
    if set(tree) != {".gitattributes", "README.md", *ARCHIVES}:
        raise ValueError("HELMET dataset repository tree changed")
    if bytes_sha256(_show(data, "README.md")) != README_SHA256:
        raise ValueError("HELMET dataset card changed")
    archives = []
    for name, expected in ARCHIVES.items():
        pointer = _lfs_pointer(_show(data, name))
        tree_entry = tree[name]
        if (
            tree_entry["git_object_id"] != expected["git_object_id"]
            or tree_entry["bytes"] != expected["pointer_bytes"]
            or pointer["oid_sha256"] != expected["oid_sha256"]
            or pointer["payload_bytes"] != expected["payload_bytes"]
        ):
            raise ValueError(f"HELMET dataset archive changed: {name}")
        archives.append({**tree_entry, **pointer, "role": expected["role"]})
    after = _pack_snapshot(data)
    if after != before:
        raise ValueError("HELMET metadata audit fetched new Git objects")

    script = code / "scripts/download_data.sh"
    if file_sha256(script) != DOWNLOAD_SCRIPT_SHA256:
        raise ValueError("HELMET download script changed")
    script_text = script.read_text(encoding="utf-8")
    if "data.tar.gz" not in script_text or "data_v2.tar.gz" in script_text:
        raise ValueError("HELMET required archive selection changed")
    configs = []
    for relative in CONFIGS:
        path = code / relative
        category = Path(relative).stem.removesuffix("_short")
        configs.append(_parse_config(path, category, path.stem.endswith("_short")))
    commit = _git(
        data,
        "show",
        "-s",
        "--format=%H%n%T%n%P%n%aI%n%an%n%s",
        DATA_REVISION,
    ).splitlines()
    return {
        "contract": contract,
        "source_qualification": source,
        "dataset": {
            "repository": contract["data"]["repository"],
            "revision": commit[0],
            "tree": commit[1],
            "parents": commit[2].split() if commit[2] else [],
            "authored_at": commit[3],
            "author": commit[4],
            "subject": commit[5],
            "readme_sha256": README_SHA256,
            "license_file_present": "LICENSE" in tree,
            "partial_clone_filter": _git(data, "config", "--get", "remote.origin.partialclonefilter"),
            "archives": archives,
            "new_objects_fetched": 0,
            "lfs_payloads_materialized": False,
        },
        "download_script": {
            "path": "scripts/download_data.sh",
            "sha256": DOWNLOAD_SCRIPT_SHA256,
            "upstream_url_mode": "mutable main",
            "qualified_replacement_url": (
                "https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/"
                f"{DATA_REVISION}/data.tar.gz"
            ),
            "required_archive": "data.tar.gz",
            "excluded_archive": "data_v2.tar.gz",
        },
        "configs": configs,
    }


def prepare(args):
    contract_path = args.contract.expanduser().resolve()
    stable = _stable_audit(
        contract_path,
        args.code_checkout.expanduser().resolve(),
        args.data_checkout.expanduser().resolve(),
    )
    candidate = _device(args.candidate_device)
    root = shutil.disk_usage("/")
    report = {
        "format": "speck_helmet_data_metadata_qualification",
        "format_version": 1,
        "status": "metadata_and_storage_plan_qualified_mount_and_payload_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {"path": str(contract_path)},
        "source_qualification": stable["source_qualification"],
        "dataset": stable["dataset"],
        "download_script": stable["download_script"],
        "configs": stable["configs"],
        "dataset_families": [
            {
                "category": category,
                "sources": DATASET_FAMILIES[category],
                "license_status": "citation present; upstream terms require post-extraction verification",
            }
            for category in CATEGORIES
        ],
        "license_audit": {
            "code_license": "MIT",
            "dataset_repository_license_file_present": stable["dataset"]["license_file_present"],
            "dataset_card_license_declaration": None,
            "component_licenses_qualified": False,
            "rule": "do not infer dataset rights from the MIT benchmark-code license or citation list",
        },
        "storage": {
            "required_archive_bytes": ARCHIVES["data.tar.gz"]["payload_bytes"],
            "upstream_advertised_extracted_bytes": 34_000_000_000,
            "minimum_free_bytes_before_download": MINIMUM_VOLUME_FREE_BYTES,
            "minimum_rationale": "retain compressed archive plus extracted data, hashes, results, and failure-recovery headroom",
            "root_filesystem_free_bytes": root.free,
            "root_filesystem_qualified": root.free >= MINIMUM_VOLUME_FREE_BYTES,
            "candidate_device": candidate,
            "candidate_capacity_qualified": candidate["bytes"] >= MINIMUM_VOLUME_FREE_BYTES,
            "candidate_mounted": any(candidate["mountpoints"] or []),
            "required_mount_options": ["nodev", "nosuid", "noexec"],
            "required_next": "administrator mounts the intended separate filesystem and grants a dedicated user-owned Speck directory; rerun capacity audit before download",
        },
        "decision": {
            "metadata_qualified": True,
            "download_authorized": False,
            "extraction_authorized": False,
            "execution_authorized": False,
            "reason": "candidate device is unmounted and component dataset licenses are not yet qualified",
        },
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(args.output, report)
    return report


def check(args):
    report = json.loads(args.output.expanduser().resolve().read_text(encoding="utf-8"))
    stable = _stable_audit(
        args.contract.expanduser().resolve(),
        args.code_checkout.expanduser().resolve(),
        args.data_checkout.expanduser().resolve(),
    )
    if (
        report.get("format") != "speck_helmet_data_metadata_qualification"
        or report.get("status")
        != "metadata_and_storage_plan_qualified_mount_and_payload_blocked"
        or report.get("dataset") != stable["dataset"]
        or report.get("download_script") != stable["download_script"]
        or report.get("configs") != stable["configs"]
        or report.get("decision", {}).get("download_authorized") is not False
        or report.get("decision", {}).get("execution_authorized") is not False
    ):
        raise ValueError("HELMET metadata qualification no longer matches pinned inputs")
    candidate = _device(args.candidate_device)
    if candidate != report["storage"]["candidate_device"]:
        raise ValueError("HELMET candidate storage device state changed; run a new qualification")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_data_metadata_audit.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["runner_sha256"]:
        raise ValueError("HELMET metadata audit runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"HELMET data: {report['status']} "
        f"({report['dataset']['revision'][:12]})"
    )


if __name__ == "__main__":
    main()
