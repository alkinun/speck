"""Qualify a minimal deterministic HELMET demo-selection patch on synthetic fixtures."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--helmet-checkout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker-source", type=Path)
    parser.add_argument("--worker-loader", choices=("narrativeqa", "multi_lexsum"))
    parser.add_argument("--worker-seed", type=int)
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


def repository_revision():
    return subprocess.run(
        ["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def target_unseeded_shuffles(source):
    tree = ast.parse(source)
    target_functions = {"load_narrativeqa", "load_multi_lexsum"}
    counts = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in target_functions:
            continue
        counts[function.name] = sum(
            1
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shuffle"
            and not node.args
            and not any(keyword.arg == "seed" for keyword in node.keywords)
        )
    return counts


def runtime_identity():
    return {
        name: importlib.metadata.version(name) for name in ("datasets", "numpy", "pyarrow")
    }


def _load_patched_module(source):
    utilities = types.ModuleType("utils")
    utilities.calculate_metrics = lambda *args, **kwargs: {}
    utilities.parse_output = lambda *args, **kwargs: None
    utilities.parse_rankings = lambda *args, **kwargs: []
    utilities.calculate_retrieval_metrics = lambda *args, **kwargs: {}
    sys.modules["utils"] = utilities
    spec = importlib.util.spec_from_file_location("speck_helmet_seeded_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _narrative_fixture(datasets):
    train = datasets.Dataset.from_list(
        [
            {
                "document": {"text": f"train story {index}"},
                "question": {"text": f"train question {index}"},
                "answers": [{"text": f"train answer {index}"}],
            }
            for index in range(8)
        ]
    )
    test = datasets.Dataset.from_list(
        [
            {
                "document": {"text": f"evaluation story {index}"},
                "question": {"text": f"evaluation question {index}"},
                "answers": [{"text": f"evaluation answer {index}"}],
            }
            for index in range(3)
        ]
    )
    return datasets.DatasetDict({"train": train, "test": test})


def _multilexsum_fixture(datasets):
    train = datasets.Dataset.from_list(
        [
            {
                "sources": [f"train source {index}"],
                "summary/short": f"train summary {index}",
            }
            for index in range(8)
        ]
    )
    validation = datasets.Dataset.from_list(
        [
            {
                "sources": [f"evaluation source {index}"],
                "summary/short": f"evaluation summary {index}",
            }
            for index in range(3)
        ]
    )
    return datasets.DatasetDict({"train": train, "validation": validation})


def worker(source, loader, seed):
    import datasets

    datasets.disable_progress_bars()
    module = _load_patched_module(source)
    module.filter_length = lambda data, *args, **kwargs: data
    module.truncate_llama2 = lambda name, data, *args, **kwargs: data
    module.AutoTokenizer.from_pretrained = lambda *args, **kwargs: (
        lambda text: {"input_ids": range(131_073)}
    )
    if loader == "narrativeqa":
        fixture = _narrative_fixture(datasets)
        module.load_dataset = lambda *args, **kwargs: fixture
        result = module.load_narrativeqa(
            "narrativeqa_fixture", shots=2, max_samples=None, seed=seed
        )
    else:
        fixture = _multilexsum_fixture(datasets)
        module.load_dataset = lambda *args, **kwargs: fixture
        result = module.load_multi_lexsum(
            "multi_lexsum_fixture", shots=2, max_samples=None, seed=seed
        )
    rows = result["data"].to_list()
    prompts = [result["prompt_template"].format(**row) for row in rows]
    payload = {
        "loader": loader,
        "seed": seed,
        "rows": len(rows),
        "demo_sha256": hashlib.sha256(
            json.dumps([row["demo"] for row in rows], separators=(",", ":")).encode()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    print(json.dumps(payload, sort_keys=True))


def _worker_result(source, loader, seed, hash_seed):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(hash_seed),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-source",
            str(source),
            "--worker-loader",
            loader,
            "--worker-seed",
            str(seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        raise ValueError(
            f"HELMET {loader} fixture worker failed: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def stage_source(protocol, checkout, stage):
    source = checkout / protocol["upstream"]["target"]
    if file_sha256(source) != protocol["upstream"]["target_sha256"]:
        raise ValueError("HELMET demo-patch upstream source changed")
    staged = stage / "data.py"
    shutil.copy2(source, staged)
    patch = Path(__file__).parents[1] / protocol["patch"]["path"]
    if file_sha256(patch) != protocol["patch"]["sha256"]:
        raise ValueError("HELMET demo patch changed")
    subprocess.run(
        ["git", "-C", str(stage), "apply", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(stage), "apply", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    if file_sha256(staged) != protocol["patch"]["patched_target_sha256"]:
        raise ValueError("HELMET demo patch produced unexpected source")
    return source, staged, patch


def qualify(protocol, checkout):
    if runtime_identity() != protocol["runtime"]:
        raise ValueError("HELMET demo-patch runtime changed")
    with tempfile.TemporaryDirectory(prefix="speck-helmet-seeded-demos-") as temporary:
        source, patched, patch = stage_source(protocol, checkout, Path(temporary))
        upstream_counts = target_unseeded_shuffles(source.read_text(encoding="utf-8"))
        patched_counts = target_unseeded_shuffles(patched.read_text(encoding="utf-8"))
        if upstream_counts != {"load_narrativeqa": 1, "load_multi_lexsum": 1}:
            raise ValueError("HELMET upstream unseeded-demo diagnosis changed")
        if patched_counts != {"load_narrativeqa": 0, "load_multi_lexsum": 0}:
            raise ValueError("HELMET demo patch did not remove both unseeded selections")
        fixtures = protocol["fixtures"]
        results = {}
        for loader in fixtures["loaders"]:
            primary = [
                _worker_result(
                    patched,
                    loader,
                    fixtures["primary_seed"],
                    hash_seed,
                )
                for hash_seed in fixtures["process_hash_seeds"]
            ]
            contrast = _worker_result(
                patched,
                loader,
                fixtures["contrast_seed"],
                fixtures["process_hash_seeds"][0],
            )
            if primary[0] != primary[1]:
                raise ValueError(f"HELMET {loader} prompts differ across processes")
            if primary[0]["demo_sha256"] == contrast["demo_sha256"]:
                raise ValueError(f"HELMET {loader} seed does not control demonstrations")
            if (
                primary[0] != fixtures["expected"][loader]["primary"]
                or contrast != fixtures["expected"][loader]["contrast"]
            ):
                raise ValueError(f"HELMET {loader} fixture identity changed")
            results[loader] = {
                "primary": primary[0],
                "cross_process_identical": True,
                "contrast": contrast,
                "seed_controls_demonstrations": True,
            }
        return {
            "upstream_target_sha256": file_sha256(source),
            "patch_sha256": file_sha256(patch),
            "patched_target_sha256": file_sha256(patched),
            "upstream_unseeded_shuffles": upstream_counts,
            "patched_unseeded_shuffles": patched_counts,
            "loaders": results,
        }


def prepare(protocol, protocol_path, checkout):
    qualification = qualify(protocol, checkout)
    return {
        "format": "speck_helmet_seeded_demos_qualification",
        "format_version": 1,
        "status": "two_loader_seeded_demo_patch_qualified_on_fixtures",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "spec_identity_sha256": protocol_identity(protocol),
        },
        "runtime": runtime_identity(),
        "qualification": qualification,
        "decision": {
            "patch_qualified": True,
            "real_dataset_prompts_qualified": False,
            "rights_blockers_changed": False,
            "current_manifest_changed": False,
            "candidate_execution_authorized": False,
        },
        "non_claims": protocol["reporting"]["non_claims"],
        "runner_revision": repository_revision(),
        "runner_sha256": file_sha256(__file__),
    }


def check(protocol, report_path, checkout):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    current = qualify(protocol, checkout)
    if (
        report.get("format") != "speck_helmet_seeded_demos_qualification"
        or report.get("status") != "two_loader_seeded_demo_patch_qualified_on_fixtures"
        or report.get("protocol", {}).get("spec_identity_sha256")
        != protocol_identity(protocol)
        or report.get("qualification") != current
        or report.get("decision", {}).get("patch_qualified") is not True
        or report.get("decision", {}).get("candidate_execution_authorized") is not False
    ):
        raise ValueError("HELMET seeded-demo qualification report is invalid")
    runner = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).parents[1]),
            "show",
            f"{report['runner_revision']}:scripts/helmet_seeded_demos_qualify.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(runner).hexdigest() != report["runner_sha256"]:
        raise ValueError("HELMET seeded-demo qualification runner changed")
    return report


def main(argv=None):
    args = arguments(argv)
    if args.worker_source is not None:
        worker(args.worker_source, args.worker_loader, args.worker_seed)
        return
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "speck_helmet_seeded_demos_protocol":
        raise ValueError("HELMET seeded-demo protocol has the wrong format")
    checkout = args.helmet_checkout.expanduser().resolve()
    if args.check:
        if protocol.get("status") not in {"frozen_unexecuted", "executed_qualified"}:
            raise ValueError("HELMET seeded-demo protocol has an invalid checked status")
        report = check(protocol, args.output.expanduser().resolve(), checkout)
    else:
        if protocol.get("status") != "frozen_unexecuted":
            raise ValueError("HELMET seeded-demo protocol must be frozen and unexecuted")
        report = prepare(protocol, protocol_path, checkout)
        atomic_json(args.output, report)
    print(f"HELMET seeded demos: {report['status']}")


if __name__ == "__main__":
    main()
