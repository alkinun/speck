import json
from pathlib import Path

from speck.model import Config
from speck.search.architecture import architecture_hash, kv_bytes_per_token, parameter_count
from speck.search.spec import SearchSettings, deterministic_seed, objective_names


experiment = Path(__file__).parents[1] / "experiments" / "speck00-200m"


def test_v2_architecture_identity_is_stable():
    config = Config.from_dict(json.loads((experiment / "model.json").read_text()))
    assert architecture_hash(config) == (
        "34b27a5b04185ece71720597cb1aa85c3b1b53bbd32dd81cb3f15c89162c54ff"
    )
    assert parameter_count(config) == 182_206_848
    assert kv_bytes_per_token(config) == 3_328


def test_v2_seed_identity_is_stable():
    assert deterministic_seed(42, "architecture", 0, 0) == 192_123_501_389_243_075
    assert deterministic_seed(42, "trial", 2, 0) == 6_674_701_210_661_975_869


def test_v2_objective_contract_is_stable():
    settings = SearchSettings.from_dict(
        json.loads((experiment / "search.json").read_text())
    )
    assert objective_names(settings) == (
        "quality.validation_nll.main",
        "memory.kv_cache_bytes_per_token",
        "memory.quantized_weight_bytes",
        "prefill.ms.context_512",
        "decode.ms_per_token.context_512",
        "memory.inference_peak_bytes.context_512",
        "prefill.ms.context_2048",
        "decode.ms_per_token.context_2048",
        "memory.inference_peak_bytes.context_2048",
    )
