from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from scripts import base_checkpoint_export
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    RoutedSwiGLUSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM


def tiny_moe():
    dense = BlockConfig(
        8,
        (
            StageConfig((AttentionSpec(4, 1),)),
            StageConfig((SwiGLUSpec(16),)),
        ),
    )
    routed = BlockConfig(
        8,
        (
            StageConfig((AttentionSpec(4, 1),)),
            StageConfig((RoutedSwiGLUSpec(8, 4, 2),)),
        ),
    )
    config = ArchitectureConfig(
        (BlockGroup(dense), BlockGroup(routed)),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=16,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    return model


def test_self_contained_moe_export_loads_with_parity_and_padding(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    try:
        template = Path(
            snapshot_download(
                repo_id=base_checkpoint_export.TEMPLATE_REPO,
                revision=base_checkpoint_export.TEMPLATE_REVISION,
                allow_patterns=list(base_checkpoint_export.TEMPLATE_FILES),
                local_files_only=True,
            )
        )
    except Exception as error:
        pytest.skip(f"pinned local Transformers template is unavailable: {error}")
    monkeypatch.setattr(
        base_checkpoint_export,
        "snapshot_download",
        lambda **kwargs: str(template),
    )
    torch.manual_seed(31)
    native = tiny_moe()
    metadata = {
        "config": native.config.settings(),
        "resolved": {
            "parameters": native.parameter_count(),
            "active_parameters": native.active_parameter_count(),
        },
    }
    output = tmp_path / "export"

    base_checkpoint_export.export(native.state_dict(), output, metadata, {"test": True})
    base_checkpoint_export.validate_moe_export(output, metadata)
    parity = base_checkpoint_export.validate_parity(
        output, native.state_dict(), metadata
    )

    assert parity["passed"]
    assert parity["parameters"] == native.parameter_count()
    state = load_file(output / "model.safetensors")
    assert state["cores.group_1_repeat_0.stages.1.branches.0.operation.router.weight"].dtype == (
        torch.float32
    )
    assert state["cores.group_1_repeat_0.stages.1.branches.0.operation.gate_proj"].dtype == (
        torch.bfloat16
    )
    exported = transformers.AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, dtype=torch.bfloat16
    ).eval()
    first = torch.tensor([[1, 3, 4, 2]])
    padded = torch.tensor([[1, 3, 4, 2, 0, 0], [1, 5, 6, 7, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])

    with torch.no_grad():
        expected = exported(input_ids=first, use_cache=False).logits
        actual = exported(
            input_ids=padded,
            attention_mask=mask,
            use_cache=False,
        ).logits

    torch.testing.assert_close(actual[0, :4], expected[0])
