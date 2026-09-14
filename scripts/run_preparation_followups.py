"""Run a finite, bound local stock/token sequence after successful source jobs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.stock_tokens import load_stock_token_plan, tokenize_stock, verify_token_stock
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def completed_stock(step, result):
    """Require the expected execution and preparation gates before tokenization."""
    if (
        result.get("plan") != step["plan"]
        or result.get("repository_revision") != step["repository_revision"]
        or result.get("format") != step["result_format"]
        or result.get("training_authority") is not False
        or result.get("capacity_target_pass") is not True
        or result.get("storage_gate_pass") is not True
        or result.get("reference_capacity", {}).get("tokens", 0) < step["target_tokens"]
        or result.get("analysis", {}).get("exact_reference_overlap") != 0
        or result.get("analysis", {}).get("reference_outputs_preserved") is not True
    ):
        raise ValueError("completed stock failed its bound execution/capacity/exclusion gates")
    controls = result["analysis"].get("controls", {})
    if (
        controls.get("firewall_exact_control", {}).get("reason") != "exact_duplicate"
        or controls.get("firewall_near_control", {}).get("reason") != "near_duplicate"
    ):
        raise ValueError("completed stock lacks its exact/near positive controls")


def wait_for_source(service, *, deadline, result_path=None):
    while True:
        values = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                service,
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        )
        state = dict(line.split("=", 1) for line in values.splitlines() if "=" in line)
        if (
            state.get("LoadState") == "not-found"
            and state.get("ActiveState") == "inactive"
            and result_path is not None
            and Path(result_path).is_file()
        ):
            # systemd garbage-collects successful transient services. Their default
            # exit fields no longer prove anything; the caller must validate the
            # complete hash-bound publication before doing dependent work.
            return {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "completion_basis": "published_result_requires_bound_validation",
                "service_exit_status_available": False,
            }
        if state.get("LoadState") != "loaded":
            raise ValueError(f"required preparation service is not loaded: {service}")
        if state["ActiveState"] == "inactive":
            if state.get("Result") != "success" or state.get("ExecMainStatus") != "0":
                raise ValueError(f"preparation service did not finish successfully: {service}")
            state["service_exit_status_available"] = True
            return state
        if state["ActiveState"] not in ("active", "activating", "deactivating"):
            raise ValueError(f"preparation service failed: {service}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"preparation followup deadline reached: {service}")
        time.sleep(30)


def build_tokens(step, working, tokenizer_id, revision):
    require_data_mount(working)
    result_path = Path(step["result_path"])
    result = json.loads(result_path.read_text())
    completed_stock(step, result)
    plan_path = working / f"{step['source_id']}-token-plan.json"
    token_spec = {
        "format": "speck_stock_tokenization_plan",
        "format_version": 1,
        "source_id": step["source_id"],
        "category": step["category"],
        "stock_result": {"path": str(result_path), "sha256": file_sha256(result_path)},
        "tokenizer_decision": tokenizer_id,
        "shard_tokens": 100000000,
        "output_directory": step["token_directory"],
        "training_authority": False,
    }
    if Path(step["token_result_path"]).exists():
        raise FileExistsError(step["token_result_path"])
    durable_json(plan_path, token_spec)
    loaded = load_stock_token_plan(plan_path)  # Rehash text/parent/reference inputs.
    print(f"tokenizing checked source: {step['source_id']}", flush=True)
    tokens = tokenize_stock(loaded)
    verify_token_stock(loaded["output_directory"], loaded)
    tokens.update(
        format="speck_stock_tokenization_result",
        format_version=1,
        repository_revision=revision,
        plan={"path": str(plan_path), "sha256": file_sha256(plan_path)},
        training_authority=False,
        completed_reopen_pass=True,
    )
    durable_json(step["token_result_path"], tokens)
    return tokens


def require_data_mount(path):
    disk = subprocess.check_output(
        ["findmnt", "-n", "-o", "UUID", "--target", str(path)], text=True
    ).strip()
    if disk != "b64b59d1-ea2c-4206-9171-b7cd739f3eff":
        raise ValueError("preparation data filesystem is not mounted at the expected location")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("followups require a clean frozen implementation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    spec = json.loads(args.plan.read_text())
    if (
        spec.get("format") != "speck_local_preparation_followups"
        or spec.get("format_version") != 1
        or spec.get("training_authority") is not False
        or spec.get("maximum_wait_hours") != 24
        or [s["source_id"] for s in spec["token_stocks"]] != ["finemath_4plus", "cosmopedia_v2"]
    ):
        raise ValueError("unsupported finite preparation sequence")
    working = Path(spec["working_directory"])
    require_data_mount(working.parent)
    working.mkdir(parents=True, exist_ok=False)
    execution = {
        "repository_revision": revision,
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        "training_authority": False,
    }
    durable_json(working / "execution.json", execution)
    # Every known input is bound before waiting; future result hashes are captured on completion.
    for step in spec["token_stocks"]:
        if _bound_identity(step["plan"], args.plan.parent) != step["plan"]:
            raise ValueError("followup source plan must use its frozen absolute identity")
    tokenizer_id = _bound_identity(spec["tokenizer_decision"], args.plan.parent)
    web = spec["fineweb_e1s"]
    _bound_identity(web["plan"], args.plan.parent)
    deadline = time.monotonic() + spec["maximum_wait_hours"] * 3600
    done = []
    try:
        for step in spec["token_stocks"]:
            print(f"waiting for completed source: {step['source_id']}", flush=True)
            service_state = wait_for_source(
                step["service"], deadline=deadline, result_path=step["result_path"]
            )
            build_tokens(step, working, tokenizer_id, revision)
            done.append(
                {
                    "source_id": step["source_id"],
                    "service_completion": service_state,
                    "token_result": {
                        "path": step["token_result_path"],
                        "sha256": file_sha256(step["token_result_path"]),
                    },
                }
            )
            durable_json(working / "progress.json", {**execution, "completed": done})
        if Path(web["result_path"]).exists():
            raise FileExistsError(web["result_path"])
        _bound_identity(web["plan"], args.plan.parent)
        require_data_mount(working)
        print("starting bound FineWeb-Edu E1S text tranche", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.prepare_fineweb_edu_stock",
                web["plan"]["path"],
                web["result_path"],
            ],
            cwd=root,
            check=True,
        )
        build_tokens({**web, "repository_revision": revision}, working, tokenizer_id, revision)
        durable_json(
            working / "complete.json",
            {
                **execution,
                "completed": done,
                "fineweb_result": {
                    "path": web["result_path"],
                    "sha256": file_sha256(web["result_path"]),
                },
                "fineweb_token_result": {
                    "path": web["token_result_path"],
                    "sha256": file_sha256(web["token_result_path"]),
                },
                "boundary": "Three checked source-specific token caches completed. Joint dataset eligibility and launch manifests remain separate; no training launch.",
            },
        )
    except BaseException as error:
        durable_json(
            working / "failure.json",
            {
                **execution,
                "completed": done,
                "error_type": type(error).__name__,
                "error": str(error),
                "boundary": "Dependent followups stopped. Original jobs, failed outputs and attempts remain preserved. No automatic restart.",
            },
        )
        raise


if __name__ == "__main__":
    main()
