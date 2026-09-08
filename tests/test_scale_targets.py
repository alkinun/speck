import hashlib
import json
from pathlib import Path

import pytest
import torch

from speck.model import build_model
from speck.scale_targets import load_and_generate, model_settings

ROOT = Path(__file__).parents[1]
TARGETS = ROOT / "research" / "flagship" / "targets"
SPEC = TARGETS / "scale-targets-v2.json"
ACCOUNTING = TARGETS / "accounting-v2.json"
LEGACY_SPEC = TARGETS / "scale-targets-v1.json"
LEGACY_ACCOUNTING = TARGETS / "accounting-v1.json"


def test_committed_scale_accounting_is_current_and_exact():
    generated = load_and_generate(SPEC, ROOT)
    committed = json.loads(ACCOUNTING.read_text())

    assert generated == committed
    assert generated["source_spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert {
        target["id"]: target["parameter_accounting"]["total_parameters"]
        for target in generated["targets"]
    } == {
        "ladder-60m": 60_561_712,
        "ladder-150m": 153_960_858,
        "ladder-220m": 220_786_016,
        "ladder-350m": 351_030_008,
        "flagship-600m": 592_344_884,
        "ladder-750m": 743_449_560,
        "flagship-1.2b": 1_195_884_576,
    }
    assert all(
        target["active_parameters"] == target["parameter_accounting"]["total_parameters"]
        for target in generated["targets"]
    )


def test_v1_scale_evidence_remains_regenerable_and_hash_bound_by_v2():
    spec = json.loads(SPEC.read_text())
    predecessor_spec = spec["supersedes"]["scale_target_spec_v1"]
    predecessor_accounting = spec["supersedes"]["scale_accounting_v1"]

    assert hashlib.sha256(LEGACY_SPEC.read_bytes()).hexdigest() == predecessor_spec["sha256"]
    assert (
        hashlib.sha256(LEGACY_ACCOUNTING.read_bytes()).hexdigest()
        == predecessor_accounting["sha256"]
    )
    assert load_and_generate(LEGACY_SPEC, ROOT) == json.loads(LEGACY_ACCOUNTING.read_text())


def test_every_compact_geometry_builds_and_matches_analytic_flops():
    spec = json.loads(SPEC.read_text())
    accounting = {target["id"]: target for target in json.loads(ACCOUNTING.read_text())["targets"]}
    vocab_size = spec["tokenizer_fallback"]["effective_vocab_size"]

    for target in spec["targets"]:
        expected = accounting[target["id"]]
        settings = model_settings(target, vocab_size)
        settings["expected_parameters"] = expected["parameter_accounting"]["total_parameters"]
        settings["expected_active_parameters"] = expected["active_parameters"]
        with torch.device("meta"):
            model = build_model(settings, vocab_size)
        assert model.parameter_count() == expected["parameter_accounting"]["total_parameters"]
        assert model.active_parameter_count() == expected["active_parameters"]
        assert [
            model.flops_per_token(point["length"]) for point in expected["flop_accounting"]
        ] == [point["analytic_training_flops_per_token"] for point in expected["flop_accounting"]]


def test_state_and_optimizer_estimates_keep_precision_and_six_nd_explicit():
    targets = json.loads(ACCOUNTING.read_text())["targets"]
    for target in targets:
        optimizer = target["optimizer_state_estimate"]
        roles = optimizer["roles"]
        muon = roles["muon"]["parameters"]
        adam_parameters = roles["adamw_decay"]["parameters"] + roles["adamw_no_decay"]["parameters"]
        adam_tensors = roles["adamw_decay"]["tensors"] + roles["adamw_no_decay"]["tensors"]
        assert optimizer["native_bf16_state_bytes"] == (
            2 * muon + 4 * adam_parameters + 4 * adam_tensors
        )
        assert target["six_nd_flops_per_token"] == (
            6 * target["parameter_accounting"]["total_parameters"]
        )
        state = target["state_geometry"]
        assert state["fixed_recurrent_state_bytes"] == (
            state["recurrent_matrix_bytes"] + state["convolution_bytes"]
        )
        for point in state["points"]:
            assert (
                point["instantiated_state"]["bf16"]["by_kind"]["kimi_delta_attention"]
                == state["fixed_recurrent_state_bytes"]
            )


def test_tokenizer_fallback_is_explicit_and_targets_cannot_launch():
    spec = json.loads(SPEC.read_text())
    accounting = json.loads(ACCOUNTING.read_text())

    assert spec["tokenizer_fallback"]["selection_status"] == "D5_pending_fallback_only"
    assert spec["tokenizer_fallback"]["effective_vocab_size"] == 32_003
    assert accounting["launch_authority"] is False
    assert accounting["requires_data_launch_authority"] is True
    assert all(target["status"].endswith("not_launchable") for target in accounting["targets"])
    for target in accounting["targets"]:
        fallback = target["tokenizer_embedding_and_head_costs"][0]
        assert fallback["physical_tied_embedding_and_lm_head_parameters"] == (
            fallback["effective_vocab_size_with_chat_tokens"] * target["embedding_size"]
        )
        membership = target["optimizer_state_estimate"]["embedding_head_membership"]
        assert membership["physical_parameter_objects"] == 1
        assert membership["optimizer_memberships"] == 1
        assert membership["optimizer_role"] == "adamw_no_decay"
        for point in target["flop_accounting"]:
            assert point["embedding_lookup_flops_per_token"] == 0
            assert point["lm_head_projection_training_flops_per_token"] == (
                6 * fallback["physical_tied_embedding_and_lm_head_parameters"]
            )
    for forbidden in spec["forbidden_files"]:
        assert not (TARGETS / forbidden).exists()


def test_historical_shape_a_hashes_are_immutable_successor_inputs(tmp_path):
    spec = json.loads(SPEC.read_text())
    historical = spec["supersedes"]["legacy_shape_a"]["target"]
    altered = json.loads(SPEC.read_text())
    altered["supersedes"]["legacy_shape_a"]["target"][1] = "0" * 64
    path = tmp_path / "scale-targets-v2.json"
    path.write_text(json.dumps(altered))

    assert historical[0].endswith("shape-a/target.json")
    with pytest.raises(ValueError, match="historical Shape-A identity changed"):
        load_and_generate(path, ROOT)
