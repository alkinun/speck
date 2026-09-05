import shlex
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_baseline_analyze import arguments as analyze_arguments
from scripts.paper_proxy_launch_qualify import arguments, load_object, validate_frozen_inputs

root = Path(__file__).parents[1]
contract_path = root / "research/paper-1/proxy_launch_v1.json"


def test_proxy_launch_contract_matches_all_frozen_inputs():
    result = validate_frozen_inputs(load_object(contract_path), root)

    assert set(result["runs"]) == set(
        result["values"]["storage_qualification"]["operational_binding"]["runs"][
            index
        ]["run"]
        for index in range(6)
    )


def test_proxy_launch_contract_rejects_prerequisite_drift():
    contract = deepcopy(load_object(contract_path))
    contract["prerequisites"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="prerequisite drifted"):
        validate_frozen_inputs(contract, root)


def test_proxy_launch_target_lock_command_names_all_dense_controls():
    contract = load_object(contract_path)
    command = shlex.split(contract["execution"]["target_lock"]["command"])
    module_index = command.index("scripts.paper_baseline_analyze")
    parsed = analyze_arguments(command[module_index + 1 :])

    assert parsed.command == "lock-target"
    assert [path.stem for path in parsed.results] == contract["execution"]["control_runs"]


def test_proxy_launch_arguments_require_an_output():
    args = arguments(["contract.json", "--output", "result.json"])

    assert args.contract == Path("contract.json")
    assert args.output == Path("result.json")
