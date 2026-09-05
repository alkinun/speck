"""Audit pinned NoLiMa license and dataset metadata without materializing payload files."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from speck.external import qualify_external_suite, validate_external_suite

DATA_REVISION = "378115b1f136b6ba78f90f78682bc55f70ec3ddd"
ADOBE_LICENSE_SHA256 = "8638b5a5beb5e1cdf06a09512d23268358b64f646775581d8276e47329e0aa06"
HAYSTACK_LICENSES_SHA256 = "0a295ce17544dc60964bfb7d817388c6f775680c2f4014dc991f9efeb5400b42"
README_SHA256 = "6ea416e927553f5acfc0bdbd95306a2ef98a21d02d53f78e9befea0f10318844"
LFS_BOOK_ARCHIVE = {
    "path": "haystack/books.tar.gz",
    "oid_sha256": "43daa2c4c11c608743129cbd688cb5ab06245872a0551fb98d33f7a061220266",
    "bytes": 10_023_582,
}
RESTRICTED_PATHS = (
    *(f"haystack/rand_shuffle/rand_book_{index}.txt" for index in range(1, 6)),
    *(f"haystack/rand_shuffle_long/rand_book_{index}.txt" for index in range(1, 6)),
    "needlesets/needle_set.json",
    "needlesets/needle_set_MC.json",
    "needlesets/needle_set_ONLYDirect.json",
    "needlesets/needle_set_hard.json",
    "needlesets/needle_set_w_CoT.json",
    "needlesets/needle_set_w_Distractor.json",
)
HAYSTACK_WORKS = (
    {
        "source": "Little Brother — Cory Doctorow",
        "license": "CC-BY-NC-SA-3.0",
        "commercial_use": False,
        "share_alike": True,
    },
    {
        "source": "Zero Sum Game — S. L. Huang",
        "license": "CC-BY-NC-SA-4.0",
        "commercial_use": False,
        "share_alike": True,
    },
    {
        "source": "Life Blood — Thomas Hoover",
        "license": "CC-BY-3.0",
        "commercial_use": True,
        "share_alike": False,
    },
    {
        "source": "Overclocked — Cory Doctorow",
        "license": "CC-BY-NC-SA-2.5",
        "commercial_use": False,
        "share_alike": True,
    },
    {
        "source": "Root of Unity — S. L. Huang",
        "license": "CC-BY-NC-SA-4.0",
        "commercial_use": False,
        "share_alike": True,
    },
    {
        "source": "With a Little Help — Cory Doctorow",
        "license": "CC-BY-NC-SA-3.0",
        "commercial_use": False,
        "share_alike": True,
    },
    {
        "source": "Rebecca of Sunnybrook Farm — Kate Douglas Wiggin",
        "license": "public-domain-US",
        "commercial_use": True,
        "share_alike": False,
    },
    {
        "source": "The Thing Beyond Reason — Elisabeth Sanxay Holding",
        "license": "public-domain-US",
        "commercial_use": True,
        "share_alike": False,
    },
    {
        "source": "If Then Else — Barbara Fister",
        "license": "CC-BY-NC-4.0",
        "commercial_use": False,
        "share_alike": False,
    },
    {
        "source": "A Vessel for Offering — Darren R. Hawkins",
        "license": "CC-BY-3.0-US",
        "commercial_use": True,
        "share_alike": False,
    },
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--code-checkout", type=Path, required=True)
    parser.add_argument("--data-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
            raise ValueError(f"cannot parse NoLiMa tree entry: {line}")
        mode, kind, object_id, size, path = match.groups()
        entries[path] = {
            "path": path,
            "mode": mode,
            "type": kind,
            "git_object_id": object_id,
            "bytes": None if size == "-" else int(size),
        }
    return entries


def _local_objects(checkout):
    objects = set()
    pack_dir = (
        Path(checkout) / "objects/pack"
        if (Path(checkout) / "objects").is_dir()
        else Path(checkout) / ".git/objects/pack"
    )
    for index in sorted(pack_dir.glob("*.idx")):
        output = subprocess.run(
            ["git", "verify-pack", "-v", str(index)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 2 and re.fullmatch(r"[0-9a-f]{40}", fields[0]):
                objects.add(fields[0])
    return objects


def _pack_snapshot(checkout):
    git_dir = Path(checkout) if (Path(checkout) / "objects").is_dir() else Path(checkout) / ".git"
    pack_dir = git_dir / "objects/pack"
    entries = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(pack_dir.iterdir())
        if path.is_file()
    ]
    return {"files": entries, "sha256": bytes_sha256(json.dumps(entries, sort_keys=True).encode())}


def _show(checkout, path):
    return _git(checkout, "show", f"{DATA_REVISION}:{path}", binary=True)


def _lfs_pointer(value):
    text = value.decode("utf-8")
    match = re.fullmatch(
        r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n",
        text,
    )
    if match is None:
        raise ValueError("NoLiMa books archive is not a canonical Git-LFS pointer")
    return {"oid_sha256": match.group(1), "bytes": int(match.group(2))}


def _stable_audit(contract_path, code_checkout, data_checkout):
    contract = validate_external_suite(contract_path)
    if contract["suite_id"] != "nolima" or contract["data"]["revision"] != DATA_REVISION:
        raise ValueError("NoLiMa license audit received a different contract")
    source = qualify_external_suite(
        contract_path, {contract["upstream"]["id"]: code_checkout}
    )
    if _git(data_checkout, "cat-file", "-t", DATA_REVISION) != "commit":
        raise ValueError("NoLiMa dataset revision is unavailable locally")
    before = _pack_snapshot(data_checkout)
    tree = _tree(data_checkout)
    missing = sorted(set(RESTRICTED_PATHS) - set(tree))
    if missing:
        raise ValueError(f"NoLiMa dataset tree is missing declared payloads: {missing}")
    code_license = (Path(code_checkout) / "LICENSE").read_bytes()
    data_license = _show(data_checkout, "LICENSE")
    licenses = _show(data_checkout, "haystack/LICENSES.md")
    readme = _show(data_checkout, "README.md")
    if (
        bytes_sha256(code_license) != ADOBE_LICENSE_SHA256
        or bytes_sha256(data_license) != ADOBE_LICENSE_SHA256
        or code_license != data_license
        or bytes_sha256(licenses) != HAYSTACK_LICENSES_SHA256
        or bytes_sha256(readme) != README_SHA256
    ):
        raise ValueError("NoLiMa license or dataset card changed")
    pointer = _lfs_pointer(_show(data_checkout, LFS_BOOK_ARCHIVE["path"]))
    if pointer != {key: LFS_BOOK_ARCHIVE[key] for key in ("oid_sha256", "bytes")}:
        raise ValueError("NoLiMa optional LFS book archive changed")
    local_objects = _local_objects(data_checkout)
    restricted = [tree[path] for path in RESTRICTED_PATHS]
    cached = [entry for entry in restricted if entry["git_object_id"] in local_objects]
    after = _pack_snapshot(data_checkout)
    if after != before:
        raise ValueError("NoLiMa metadata audit fetched new Git objects")
    commit = _git(
        data_checkout,
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
            "partial_clone_filter": _git(data_checkout, "config", "--get", "remote.origin.partialclonefilter"),
            "bare_repository": _git(data_checkout, "rev-parse", "--is-bare-repository") == "true",
            "metadata_files": [
                {**tree[path], "sha256": digest}
                for path, digest in (
                    ("LICENSE", ADOBE_LICENSE_SHA256),
                    ("README.md", README_SHA256),
                    ("haystack/LICENSES.md", HAYSTACK_LICENSES_SHA256),
                )
            ],
            "restricted_payloads": restricted,
            "restricted_payload_bytes": sum(entry["bytes"] for entry in restricted),
            "optional_lfs_book_archive": {**tree[LFS_BOOK_ARCHIVE["path"]], **pointer},
        },
        "cache_audit": {
            "pack_snapshot_sha256": before["sha256"],
            "pack_files": before["files"],
            "new_objects_fetched": 0,
            "restricted_blobs_cached": len(cached),
            "restricted_blobs_declared": len(restricted),
            "cached_restricted_paths": [entry["path"] for entry in cached],
            "working_tree_materialized": False,
        },
    }


def prepare(args):
    contract_path = args.contract.expanduser().resolve()
    stable = _stable_audit(
        contract_path,
        args.code_checkout.expanduser().resolve(),
        args.data_checkout.expanduser().resolve(),
    )
    report = {
        "format": "speck_nolima_license_decision",
        "format_version": 1,
        "status": "metadata_and_license_audited_authorized_acceptance_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(contract_path),
            "license_spec_sha256": bytes_sha256(
                json.dumps(
                    {
                        "upstream": stable["contract"]["upstream"],
                        "benchmark": stable["contract"]["benchmark"],
                        "data_repository": stable["contract"]["data"]["repository"],
                        "data_revision": stable["contract"]["data"]["revision"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
        },
        "source_qualification": stable["source_qualification"],
        "dataset": stable["dataset"],
        "cache_audit": stable["cache_audit"],
        "adobe_research_license": {
            "sha256": ADOBE_LICENSE_SHA256,
            "acceptance_trigger": "exercising rights under the license",
            "entity_authority_required": True,
            "permitted_purpose": "academic research and teaching only",
            "commercial_use_permitted": False,
            "commercial_product_development_permitted": False,
            "activities_resulting_in_commercial_gain_permitted": False,
            "redistribution": "noncommercial research only; recipients must receive the Adobe Research License",
            "attribution": "retain all copyright notices and disclaimers",
            "revocable": True,
            "automatic_termination_on_material_breach": True,
        },
        "haystack_rights": {
            "source_file_sha256": HAYSTACK_LICENSES_SHA256,
            "works": list(HAYSTACK_WORKS),
            "noncommercial_works": sum(not work["commercial_use"] for work in HAYSTACK_WORKS),
            "share_alike_works": sum(work["share_alike"] for work in HAYSTACK_WORKS),
            "unresolved_mapping": "the license file identifies ten source works but does not map the five shuffled or five long-shuffled outputs to individual works",
            "compatibility_risk": "multiple BY-NC-SA versions appear in derived shuffled haystacks; no local legal conclusion is made about cross-license compatibility",
        },
        "decision": {
            "status": "authorized_acceptance_required",
            "download_or_use_authorized": False,
            "raw_payload_git_redistribution_authorized": False,
            "required_attestation": "An authorized representative must confirm that the entity accepts the pinned Adobe Research License solely for noncommercial academic research/teaching, authorizes local evaluation-data use, and forbids redistribution through Speck's MIT artifacts.",
            "if_not_accepted": "remove NoLiMa from required evaluation policy through a new reviewed contract version and substitute a legally compatible independent benchmark",
        },
        "interpretation_boundary": "research compliance inventory only; not legal advice",
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
        report.get("format") != "speck_nolima_license_decision"
        or report.get("status")
        != "metadata_and_license_audited_authorized_acceptance_blocked"
        or report.get("dataset") != stable["dataset"]
        or report.get("cache_audit") != stable["cache_audit"]
        or report.get("decision", {}).get("download_or_use_authorized") is not False
        or report.get("decision", {}).get("raw_payload_git_redistribution_authorized") is not False
    ):
        raise ValueError("NoLiMa license decision no longer matches pinned metadata")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/nolima_license_audit.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != report["runner_sha256"]:
        raise ValueError("NoLiMa license audit runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"NoLiMa license: {report['status']} "
        f"({report['dataset']['revision'][:12]})"
    )


if __name__ == "__main__":
    main()
