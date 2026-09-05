import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.cuda_decode_diagnostic import (
    compare_full_and_cached,
    load_contract,
    logit_metrics,
    tensor_metrics,
)
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM

root = Path(__file__).parents[1]
contract_path = root / "research" / "paper-1" / "cuda_decode_diagnostic.json"


def test_checked_cuda_decode_contract_is_frozen_and_pinned():
    _, contract, _, matrix = load_contract(contract_path)

    assert contract["seeds"] == [42, 43, 44]
    assert contract["lengths"] == [8, 64, 512]
    assert [value["id"] for value in contract["execution_matrices"]] == [
        "native_cuda",
        "torch_kda_cuda",
        "torch_kda_no_conv_history_cuda",
        "cpu_sentinel",
    ]
    assert len(matrix["planned_primary_baselines"]["arms"]) == 2


def test_cuda_decode_contract_rejects_posthoc_tolerance_change(tmp_path):
    copied = tmp_path / "research" / "paper-1"
    copied.mkdir(parents=True)
    value = deepcopy(json.loads(contract_path.read_text(encoding="utf-8")))
    value["baseline_matrix"] = str(root / value["baseline_matrix"])
    value["trigger_result"] = str(root / value["trigger_result"])
    value["numerical_contract"]["absolute_tolerance"] = 0.1
    (copied / contract_path.name).write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="tolerance"):
        load_contract(copied / contract_path.name)


def test_tensor_metrics_report_scale_shape_and_direction():
    expected = torch.tensor([[1.0, 2.0, 3.0]])
    actual = torch.tensor([[1.0, 2.1, 2.9]])

    result = tensor_metrics(actual, expected, rtol=0, atol=0.05)

    assert result["maximum_absolute_error"] == pytest.approx(0.1)
    assert result["mismatched_elements"] == 2
    assert result["total_elements"] == 3
    assert result["cosine_similarity"] < 1
    assert result["norm_ratio"] > 0
    assert result["passed"] is False


def test_logit_metrics_preserve_token_disagreement_positions():
    expected = torch.tensor([[[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]]])
    actual = torch.tensor([[[2.0, 1.0], [3.0, 2.0], [3.0, 0.0]]])

    result = logit_metrics(actual, expected, rtol=0, atol=0.5)

    assert result["argmax_agreement"] == pytest.approx(2 / 3)
    assert result["argmax_disagreements"] == 1
    assert result["argmax_disagreement_positions"] == [1]
    assert result["final_token_argmax_agreement"] is True


def test_layerwise_capture_matches_a_tiny_cpu_model():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((AttentionSpec(4, 1),)),
                        StageConfig((SwiGLUSpec(16),)),
                    ),
                )
            ),
        ),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    torch.manual_seed(7)
    model = SpeckForCausalLM(config)
    model.init_weights()
    model.eval()
    tokens = torch.randint(0, 16, (1, 8))

    report = compare_full_and_cached(model, tokens, rtol=0.02, atol=0.02)

    assert report["logits"]["passed"]
    assert report["first_failure"] is None
    assert any(module["type"] == "linear" for module in report["modules"])
    assert any(module["type"] == "rms_norm" for module in report["modules"])
    assert any(module["type"] == "operation" for module in report["modules"])
    assert any(module["type"] == "stage_residual" for module in report["modules"])
