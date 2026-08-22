import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM
from speck.train import lr_scale, optimization_step


def test_lr_scale():
    assert lr_scale(0, 10, 2, 0.1) == 0.5
    assert lr_scale(1, 10, 2, 0.1) == 1.0
    assert lr_scale(10, 10, 2, 0.1) == 0.1


def test_optimization_step_advances_the_loader():
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
    model = SpeckForCausalLM(config)
    model.init_weights()
    optimizer = model.optimizer()
    first = (torch.randint(0, 16, (1, 4)), torch.randint(0, 16, (1, 4)), {"batch": 0})
    second = (torch.randint(0, 16, (1, 4)), torch.randint(0, 16, (1, 4)), {"batch": 1})
    loader = iter([second])

    loss, grad_norm, next_batch = optimization_step(
        model,
        tuple(model.parameters()),
        optimizer,
        loader,
        first,
        accumulation=1,
        grad_clip=1.0,
        lr=1e-3,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)
    assert next_batch[2] == {"batch": 1}
    assert optimizer.param_groups[0]["lr"] == 1e-3
