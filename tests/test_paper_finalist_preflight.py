from pathlib import Path

from scripts.paper_finalist_preflight import arguments, arm_specs, load_inputs

root = Path(__file__).parents[1]
contract = root / "research" / "paper-1" / "finalist_materialization_v1.json"


def test_finalist_preflight_arguments_default_to_cuda():
    args = arguments([str(contract), "--output", "result.json"])
    assert args.contract == contract
    assert args.output == Path("result.json")
    assert args.device == "cuda"


def test_finalist_preflight_inputs_remain_qualified_and_training_blocked():
    value, artifacts = load_inputs(root, contract)
    assert value["decision"]["training_authorized"] is False
    assert artifacts["qualification"]["decision"]["data_windows_qualified"] is True
    assert artifacts["analysis_qualification"]["decision"][
        "analysis_implementation_qualified"
    ] is True


def test_finalist_preflight_arm_specs_match_frozen_geometry():
    value, _ = load_inputs(root, contract)
    specs = arm_specs(value)
    assert specs == [
        {
            "id": "dense_global_param_match",
            "parameters": 153_977_088,
            "flops_per_token_at_4096": 1_301_237_760,
            "device_batch_size": 4,
        },
        {
            "id": "five_cache_kda_gqa",
            "parameters": 153_958_938,
            "flops_per_token_at_4096": 1_021_601_280,
            "device_batch_size": 4,
        },
    ]
