"""Build and verify a pinned offline pytrec_eval runtime for HELMET."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PYTREC = {
    "url": "https://files.pythonhosted.org/packages/source/p/pytrec-eval/pytrec_eval-0.5.tar.gz",
    "filename": "pytrec_eval-0.5.tar.gz",
    "bytes": 15_248,
    "sha256": "d9eb4616e7d6b73bf1b5ba4c2c4916e88124e790b53fd61610c30158999e7bde",
    "root": "pytrec_eval-0.5",
    "license": "MIT",
}
TREC = {
    "url": "https://github.com/usnistgov/trec_eval/archive/v9.0.8.tar.gz",
    "filename": "trec_eval-v9.0.8.tar.gz",
    "bytes": 189_343,
    "sha256": "c3994a73103ec842e12df693749584a45814c35c36dcc15f38984bd463566ba1",
    "root": "trec_eval-9.0.8",
    "license": "copyright notices present; no repository license file; local research build only",
}
HELMET_REVISION = "af609c4d51b97fc35012099380aa889da961c42d"
BUILD_PACKAGES = {"setuptools": "80.9.0", "wheel": "0.45.1"}
SOURCE_DATE_EPOCH = "1600000000"
NETWORK_GUARD = r'''"""Deny and record all IPv4/IPv6 socket activity in this Python process."""
import json
import os
import socket

_log_path = os.environ["SPECK_NETWORK_AUDIT_LOG"]
_phase = os.environ.get("SPECK_NETWORK_AUDIT_PHASE", "qualification")
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_sendto = socket.socket.sendto


def _internet_socket(sock):
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def _record(operation, address):
    event = {
        "event": "network_attempt_denied",
        "phase": _phase,
        "operation": operation,
        "address": repr(address),
    }
    with open(_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    raise RuntimeError("Speck offline HELMET scorer guard denied network access")


def _connect(sock, address):
    if _internet_socket(sock):
        return _record("connect", address)
    return _original_connect(sock, address)


def _connect_ex(sock, address):
    if _internet_socket(sock):
        return _record("connect_ex", address)
    return _original_connect_ex(sock, address)


def _sendto(sock, data, *args):
    address = args[-1] if args else None
    if _internet_socket(sock):
        return _record("sendto", address)
    return _original_sendto(sock, data, *args)


def _create_connection(address, *args, **kwargs):
    return _record("create_connection", address)


def _getaddrinfo(host, *args, **kwargs):
    return _record("getaddrinfo", host)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.socket.sendto = _sendto
socket.create_connection = _create_connection
socket.getaddrinfo = _getaddrinfo
'''


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--helmet-checkout", type=Path, required=True)
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


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _identity(entries):
    return bytes_sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())


def directory_identity(directory):
    directory = Path(directory)
    entries = [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(path for path in directory.rglob("*") if path.is_file())
    ]
    return {"files": entries, "sha256": _identity(entries)}


def _entry(path, bundle, **values):
    return {
        **values,
        "path": Path(path).relative_to(bundle).as_posix(),
        "bytes": Path(path).stat().st_size,
        "sha256": file_sha256(path),
    }


def _fetch(spec, path):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with urllib.request.urlopen(spec["url"], timeout=60) as response, temporary.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != spec["bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError(f"HELMET scorer source changed: {spec['filename']}")


def _safe_extract_tar(archive, destination, expected_root):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"archive path escapes extraction root: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"archive contains unsupported member: {member.name}")
        handle.extractall(destination)
    extracted = destination / expected_root
    if not extracted.is_dir():
        raise ValueError(f"archive lacks expected root: {expected_root}")
    return extracted


def _guard_environment(bundle, audit_log, phase):
    guard = bundle / "network_guard"
    guard.mkdir(exist_ok=True)
    hook = guard / "sitecustomize.py"
    hook.write_text(NETWORK_GUARD, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(guard),
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "SPECK_NETWORK_AUDIT_LOG": str(audit_log),
            "SPECK_NETWORK_AUDIT_PHASE": phase,
            "CFLAGS": (
                f"-std=gnu17 -ffile-prefix-map={bundle / 'build'}=/usr/src/pytrec_eval"
            ),
            "CXXFLAGS": f"-ffile-prefix-map={bundle / 'build'}=/usr/src/pytrec_eval",
            "LDFLAGS": "-Wl,--build-id=none",
        }
    )
    return env, hook


def _guard_self_test(bundle):
    audit_log = bundle / "network-self-test.jsonl"
    audit_log.unlink(missing_ok=True)
    env, hook = _guard_environment(bundle, audit_log, "self_test")
    result = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
        env=env,
        capture_output=True,
        text=True,
    )
    events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    if result.returncode == 0 or len(events) != 1:
        raise RuntimeError("HELMET scorer network guard failed its self-test")
    return {"hook_sha256": file_sha256(hook), "operation": events[0]["operation"]}


def _prepare_source(bundle):
    build_root = bundle / "build"
    if build_root.exists():
        shutil.rmtree(build_root)
    extracted = build_root / "extracted"
    pytrec_root = _safe_extract_tar(
        bundle / "sources" / PYTREC["filename"], extracted / "pytrec", PYTREC["root"]
    )
    trec_root = _safe_extract_tar(
        bundle / "sources" / TREC["filename"], extracted / "trec", TREC["root"]
    )
    source = build_root / "source"
    shutil.copytree(pytrec_root, source)
    shutil.copytree(trec_root, source / "trec_eval")
    return source


def _build_once(bundle, repeat):
    source = _prepare_source(bundle)
    source_identity = directory_identity(source)
    output = bundle / "repeats" / str(repeat)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    audit_log = bundle / f"network-build-{repeat}.jsonl"
    audit_log.unlink(missing_ok=True)
    env, _ = _guard_environment(bundle, audit_log, f"build_{repeat}")
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
    )
    attempts = []
    if audit_log.is_file():
        attempts = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    wheels = list(output.glob("*.whl"))
    if result.returncode != 0 or len(wheels) != 1:
        raise RuntimeError(
            f"offline pytrec_eval build {repeat} failed:\n"
            + result.stdout[-4000:]
            + "\n"
            + result.stderr[-4000:]
        )
    if attempts:
        raise RuntimeError(f"pytrec_eval build attempted network access: {attempts}")
    return wheels[0], source_identity


def _extract_wheel(wheel, runtime):
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as handle:
        for name in handle.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"wheel path escapes runtime: {name}")
        handle.extractall(runtime)


def _metric_worker(runtime, helmet_checkout, bundle):
    audit_log = bundle / "network-metrics.jsonl"
    audit_log.unlink(missing_ok=True)
    env, _ = _guard_environment(bundle, audit_log, "metrics")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(runtime), str(helmet_checkout), env["PYTHONPATH"]]
    )
    code = """
import json
import pytrec_eval
from utils import calculate_retrieval_metrics

qrels = {
    "q1": {"d1": 1, "d2": 0, "d3": 1},
    "q2": {"a": 1, "b": 0},
}
results = {
    "q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0},
    "q2": {"b": 2.0, "a": 1.0},
}
metrics = calculate_retrieval_metrics(results, qrels, k_values=[1, 2, 3])
expected = {"P@1": 0.5, "P@2": 0.5, "Recall@1": 0.25, "Recall@2": 0.75, "Recall@3": 1.0, "MRR": 0.75}
if any(metrics[key] != value for key, value in expected.items()):
    raise ValueError((metrics, expected))
print(json.dumps({
    "module": pytrec_eval.__file__,
    "metrics": metrics,
    "expected_subset": expected,
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    attempts = []
    if audit_log.is_file():
        attempts = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    if result.returncode != 0:
        raise RuntimeError("HELMET scorer metric check failed:\n" + result.stderr[-4000:])
    if attempts:
        raise RuntimeError(f"HELMET scorer metric check attempted network access: {attempts}")
    return json.loads(result.stdout.strip())


def _platform():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "compiler": subprocess.run(
            ["g++", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
    }


def _versions():
    versions = {name: importlib.metadata.version(name) for name in BUILD_PACKAGES}
    if versions != BUILD_PACKAGES:
        raise ValueError(f"HELMET scorer build packages changed: {versions}")
    return versions


def prepare(args):
    bundle = args.bundle.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    helmet = args.helmet_checkout.expanduser().resolve()
    if bundle in {Path("/"), Path.home()} or len(bundle.parts) < 4:
        raise ValueError("HELMET scorer bundle path is too broad")
    if subprocess.run(
        ["git", "-C", str(helmet), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() != HELMET_REVISION:
        raise ValueError("HELMET scorer checkout revision changed")
    if subprocess.run(
        ["git", "-C", str(helmet), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("HELMET scorer checkout is dirty")
    versions = _versions()
    bundle.mkdir(parents=True, exist_ok=True)
    for spec in (PYTREC, TREC):
        _fetch(spec, bundle / "sources" / spec["filename"])
    guard = _guard_self_test(bundle)
    first, first_source = _build_once(bundle, 1)
    second, second_source = _build_once(bundle, 2)
    if first.name != second.name or file_sha256(first) != file_sha256(second):
        raise ValueError("pytrec_eval wheel is not reproducible across complete rebuilds")
    if first_source != second_source:
        raise ValueError("pytrec_eval injected source changed between rebuilds")
    wheel = bundle / "wheels" / first.name
    wheel.parent.mkdir(exist_ok=True)
    shutil.copy2(first, wheel)
    runtime = bundle / "runtime"
    _extract_wheel(wheel, runtime)
    metrics = _metric_worker(runtime, helmet, bundle)
    runtime_identity = directory_identity(runtime)
    sources = [
        _entry(
            bundle / "sources" / spec["filename"],
            bundle,
            id="pytrec_eval" if spec is PYTREC else "trec_eval",
            url=spec["url"],
            license=spec["license"],
        )
        for spec in (PYTREC, TREC)
    ]
    report = {
        "format": "speck_helmet_scorer_runtime_manifest",
        "format_version": 1,
        "status": "qualified_offline_platform_specific_local_runtime_unredistributable",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle),
        "helmet": {
            "revision": HELMET_REVISION,
            "utils_sha256": file_sha256(helmet / "utils.py"),
        },
        "sources": sources,
        "injected_source_identity_sha256": first_source["sha256"],
        "build": {
            "package_versions": versions,
            "source_date_epoch": int(SOURCE_DATE_EPOCH),
            "c_standard": "gnu17",
            "debug_path_prefix": "/usr/src/pytrec_eval",
            "complete_rebuilds": 2,
            "wheels_identical": True,
            "wheel": _entry(wheel, bundle),
            "platform": _platform(),
        },
        "runtime": {
            "path": runtime.relative_to(bundle).as_posix(),
            "identity_sha256": runtime_identity["sha256"],
            "files": runtime_identity["files"],
        },
        "metrics": metrics,
        "network_denial": {
            "mechanism": "sitecustomize IPv4/IPv6 socket interception during both builds and metric import",
            "guard_sha256": guard["hook_sha256"],
            "self_test": "denied_as_expected",
            "self_test_denied_operation": guard["operation"],
            "build_attempts": 0,
            "metric_attempts": 0,
        },
        "release_policy": "sources, wheel, and native runtime remain outside Git because trec_eval v9.0.8 has no explicit repository license file; only hashes and measurements may be published",
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }
    atomic_json(manifest_path, report)
    atomic_json(bundle / "helmet-scorer-runtime-manifest.json", report)
    return report


def check(args):
    manifest_path = args.manifest.expanduser().resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("format") != "speck_helmet_scorer_runtime_manifest"
        or value.get("status")
        != "qualified_offline_platform_specific_local_runtime_unredistributable"
        or value.get("helmet", {}).get("revision") != HELMET_REVISION
        or value.get("build", {}).get("package_versions") != _versions()
        or value.get("build", {}).get("platform") != _platform()
        or value.get("build", {}).get("complete_rebuilds") != 2
        or not value.get("build", {}).get("wheels_identical")
        or value.get("network_denial", {}).get("build_attempts") != 0
        or value.get("network_denial", {}).get("metric_attempts") != 0
    ):
        raise ValueError("HELMET scorer runtime manifest is invalid")
    bundle = Path(value["bundle_root"])
    for entry in [*value["sources"], value["build"]["wheel"]]:
        path = bundle / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"HELMET scorer runtime file changed: {entry['path']}")
    runtime = bundle / value["runtime"]["path"]
    for entry in value["runtime"]["files"]:
        path = runtime / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"HELMET scorer runtime file changed: {entry['path']}")
    identity = directory_identity(runtime)
    if identity["sha256"] != value["runtime"]["identity_sha256"]:
        raise ValueError("HELMET scorer runtime identity changed")
    current_metrics = _metric_worker(
        runtime, args.helmet_checkout.expanduser().resolve(), bundle
    )
    if current_metrics != value["metrics"]:
        raise ValueError("HELMET scorer metrics changed")
    runner_source = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{value['runner_revision']}:scripts/helmet_scorer_prepare.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if bytes_sha256(runner_source) != value["runner_sha256"]:
        raise ValueError("HELMET scorer runner source changed")
    return value


def main(argv=None):
    args = arguments(argv)
    report = check(args) if args.check else prepare(args)
    print(
        f"HELMET scorer: {report['status']} "
        f"({report['build']['wheel']['sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
