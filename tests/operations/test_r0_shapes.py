import copy
import json

import pytest

from speck.operations.r0_shapes import LENGTHS, shape_cases
from speck.provenance.repository import repository_root

ROOT = repository_root()


@pytest.fixture
def contracts():
    return tuple(
        json.loads((ROOT / path).read_text())
        for path in (
            "research/flagship/model_plan_v1.json",
            "archive/pregrant-history/research/flagship/targets/scale-targets-v2.json",
            "research/flagship/tokenizer_decision_v1.json",
        )
    )


def test_exact_models_independent_weight_blocks_ties_and_optimizer_coverage(contracts):
    cases = shape_cases(*contracts)
    assert [(c["architecture"], c["sequence_length"]) for c in cases] == [
        (kind, length) for kind in ("hybrid", "dense") for length in LENGTHS
    ]
    for case in cases:
        expected = 1195884576 if case["architecture"] == "hybrid" else 1072281600
        assert case["instantiated_parameters"] == case["active_parameters"] == expected
        assert case["independent_weight_blocks"] == 24
        assert case["physically_tied_embeddings_pass"]
        roles = case["optimizer_membership_and_state_estimate"]
        assert sum(r["parameters"] for r in roles["roles"].values()) == expected
        assert roles["embedding_head_membership"]["optimizer_memberships"] == 1
        assert roles["embedding_head_membership"]["optimizer_role"] == "adamw_no_decay"
        assert case["model_vocab_size"] == 32003 and case["synthetic_input_vocab_size"] == 32000
        assert case["reserved_ids_used_as_inputs"] is False
        assert all(
            case[key] is None
            for key in (
                "actual_gpu_peak_bytes",
                "measured_tokens_per_second",
                "gpu_fit_pass",
                "backward_pass",
                "resume_parity_pass",
                "four_gpu_ddp_pass",
            )
        )


def test_dense_control_changes_only_mixer_and_derived_parameter_counts(contracts):
    cases = shape_cases(*contracts)
    for hybrid, dense in zip(cases[:3], cases[3:], strict=True):
        a, b = copy.deepcopy(hybrid["model"]), copy.deepcopy(dense["model"])
        for settings in (a, b):
            settings.pop("expected_parameters")
            settings.pop("expected_active_parameters")
            for group in settings["blocks"]:
                mixer = group["block"]["stages"][0]["branches"][0]
                if mixer["kind"] == "attention":
                    assert mixer["rope_dim"] == 0
                    assert mixer["head_dim"] == 128 and mixer["num_key_value_heads"] == 4
                else:
                    assert mixer["output_gate_activation"] == "sigmoid"
                group["block"]["stages"][0] = "mixer removed for common-geometry comparison"
        assert a == b


def test_inference_state_sizes_match_independent_formulas_not_peak_memory(contracts):
    cases = shape_cases(*contracts)
    # KDA state remains FP32 even with BF16 parameters, convolution and global KV cache.
    recurrent = 18 * 16 * 128 * 128 * 4
    convolution = 18 * (2 * 8 * 128 + 16 * 128) * 3 * 2
    for case in cases:
        state = case["inference_state_shape_bytes_batch_one"]
        layers = 6 if case["architecture"] == "hybrid" else 24
        kv = layers * 2 * case["sequence_length"] * 4 * 128 * 2
        assert state["by_kind"]["attention_kv"] == kv
        assert state["total_bytes"] == kv + (recurrent + convolution if layers == 6 else 0)
        assert case["actual_gpu_peak_bytes"] is None


@pytest.mark.parametrize("change", ["depth", "gate", "tie", "vocabulary", "head", "count"])
def test_contract_or_geometry_drift_fails_before_shape_publication(contracts, change):
    model, retained, decision = copy.deepcopy(contracts)
    if change == "depth":
        model["flagship"]["depth"] = 28
    elif change == "gate":
        model["flagship"]["output_gate"] = "silu"
    elif change == "tie":
        model["flagship"]["physically_tied_embeddings"] = False
    elif change == "vocabulary":
        decision["reserved_role_capacity"]["effective_model_vocab_size"] = 32000
    elif change == "head":
        next(t for t in retained["targets"] if t["id"] == "flagship-1.2b")[
            "num_key_value_heads"
        ] = 2
    else:
        model["flagship"]["reference_parameters"] += 1
    with pytest.raises(ValueError):
        shape_cases(model, retained, decision)


def test_changed_bound_input_cannot_publish_result(tmp_path):
    from speck.operations.r0_shapes import prepare_r0_shapes
    from speck.provenance.io import durable_json

    spec = json.loads((ROOT / "research/flagship/r0_shape_preparation_v1.json").read_text())
    spec["inputs"]["model"]["sha256"] = "0" * 64
    path, output = tmp_path / "bad-plan.json", tmp_path / "result.json"
    durable_json(path, spec)
    with pytest.raises(ValueError, match="input identity differs"):
        prepare_r0_shapes(path, output)
    assert not output.exists()
