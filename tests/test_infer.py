import pytest

from scripts.infer import arguments, load_checkpoint_model
from speck.architecture import ArchitectureConfig, BlockConfig, BlockGroup, StageConfig, SwiGLUSpec
from speck.checkpoint import save
from speck.model import SpeckForCausalLM


def tiny_config():
    return ArchitectureConfig(
        blocks=(
            BlockGroup(
                BlockConfig(
                    hidden_size=8,
                    stages=(StageConfig((SwiGLUSpec(intermediate_size=16),)),),
                )
            ),
        ),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=32,
    )


def test_inference_loader_does_not_read_optimizer_state(tmp_path):
    config = tiny_config()
    source = SpeckForCausalLM(config)
    source.init_weights()
    save(
        tmp_path,
        3,
        source.state_dict(),
        {"unused": True},
        {"step": 3, "config": config.settings()},
    )
    (tmp_path / "optimizer_000003.pt").write_bytes(b"not a torch checkpoint")

    loaded, metadata = load_checkpoint_model(tmp_path, 3, "cpu", loss_backend="liger")

    assert metadata["step"] == 3
    assert loaded.training is False
    assert loaded.loss_backend == "liger"
    assert all(
        left.equal(right)
        for left, right in zip(
            source.state_dict().values(), loaded.state_dict().values(), strict=False
        )
    )


def test_inference_argument_parser_is_import_safe():
    args = arguments(["hello", "--device", "cpu", "--step", "3"])

    assert args.prompt == "hello"
    assert args.device == "cpu"
    assert args.step == 3


@pytest.mark.parametrize(
    "option,value",
    (
        ("--max-tokens", "0"),
        ("--max-tokens", "-1"),
        ("--temperature", "-0.1"),
        ("--temperature", "nan"),
        ("--temperature", "inf"),
        ("--top-k", "0"),
        ("--top-k", "-1"),
    ),
)
def test_invalid_sampling_arguments_fail_during_parsing(option, value):
    with pytest.raises(SystemExit) as error:
        arguments(["hello", option, value])
    assert error.value.code == 2
