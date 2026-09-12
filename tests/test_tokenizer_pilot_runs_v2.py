import json
from pathlib import Path

import pytest

from speck.tokenizer_pilot_runs_v2 import validate_flop_correction

ROOT = Path(__file__).parents[1]
CORRECTION = ROOT / "research/flagship/tokenizer_pilot_flop_correction_v1.json"


def test_v2_validator_independently_recomputes_checked_correction():
    correction = json.loads(CORRECTION.read_text())
    scale = json.loads((ROOT / correction["inputs"]["scale_spec"]["path"]).read_text())
    preflight = json.loads((ROOT / correction["inputs"]["preflight_result"]["path"]).read_text())

    result = validate_flop_correction(correction, scale, preflight)

    assert result["analytic_flop_target"] == 498_654_977_446_391_808
    assert (
        result["tokenizers"]["speck-bpe-40960-whitespace"]["fixed_flop_token_stop"] == 1_125_458_190
    )


def test_v2_validator_rejects_changed_flops_or_authority():
    correction = json.loads(CORRECTION.read_text())
    scale = json.loads((ROOT / correction["inputs"]["scale_spec"]["path"]).read_text())
    preflight = json.loads((ROOT / correction["inputs"]["preflight_result"]["path"]).read_text())
    correction["tokenizers"][1]["fixed_flop_token_stop"] += 1
    with pytest.raises(ValueError, match="stop differs"):
        validate_flop_correction(correction, scale, preflight)

    correction = json.loads(CORRECTION.read_text())
    correction["authority"]["screen_execution"] = True
    with pytest.raises(ValueError, match="authority"):
        validate_flop_correction(correction, scale, preflight)
