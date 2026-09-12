import hashlib
import json
import math
from pathlib import Path

from speck.scale_targets import flop_accounting

ROOT = Path(__file__).parents[1]
CORRECTION = ROOT / "research/flagship/tokenizer_pilot_flop_correction_v1.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_correction_binds_inputs_and_recomputes_every_stop():
    correction = json.loads(CORRECTION.read_text())
    scale_path = ROOT / correction["inputs"]["scale_spec"]["path"]
    scale = json.loads(scale_path.read_text())
    target = next(item for item in scale["targets"] if item["id"] == correction["target_id"])

    assert correction["status"] == "corrected_pre_output_accounting_v2_materialization_required"
    assert sha256(scale_path) == correction["inputs"]["scale_spec"]["sha256"]
    for identity in correction["inputs"].values():
        assert sha256(ROOT / identity["path"]) == identity["sha256"]
    plan = correction["supersedes_section"]["plan"]
    assert sha256(ROOT / plan["path"]) == plan["sha256"]

    reference = correction["reference"]
    target_flops = (
        reference["fixed_document_tokens"] * reference["analytic_training_flops_per_token"]
    )
    assert reference["analytic_flop_target"] == target_flops
    for tokenizer in correction["tokenizers"]:
        accounting = flop_accounting(
            target,
            tokenizer["vocab_size"],
            correction["sequence_length"],
            contract_version=2,
        )
        assert (
            tokenizer["analytic_training_flops_per_token"]
            == accounting["analytic_training_flops_per_token"]
        )
        expected_stop = math.ceil(target_flops / tokenizer["analytic_training_flops_per_token"])
        assert tokenizer["fixed_flop_token_stop"] == expected_stop
        expected_aligned = (
            math.ceil(
                max(tokenizer["fixed_document_tokens"], expected_stop) / correction["batch_tokens"]
            )
            * correction["batch_tokens"]
        )
        assert tokenizer["run_stop_aligned_tokens"] == expected_aligned

    assert correction["invariants"]["optimizer_boundaries_changed_from_v10"] is False
    assert correction["authority"]["screen_execution"] is False
    assert correction["authority"]["D5_opening"] is False
