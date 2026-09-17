import pytest

from speck.evaluation.inference import arguments, load_checkpoint_model, load_checkpoint_tokenizer
from speck.model import SpeckForCausalLM
from speck.model.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.training.checkpoint import save


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
    args = arguments(["hello", "--experiment", "experiment", "--device", "cpu", "--step", "3"])

    assert args.prompt == "hello"
    assert args.device == "cpu"
    assert args.step == 3


@pytest.mark.parametrize("version", [1, 2])
def test_inference_restores_explicit_chat_version_without_passing_it_to_base_loader(
    monkeypatch, version
):
    from speck.tokenization.chat import ChatTokenizer

    class Base:
        vocab_size = 16
        bos_id, eos_id = 1, 2

        def encode(self, text):
            return [3]

        def fingerprint(self):
            return "base-tokenizer"

    base = Base()

    def load_base(*, directory):
        assert directory == "prepared"
        return base

    monkeypatch.setattr("speck.evaluation.inference.get_tokenizer", load_base)
    metadata = {
        "training_phase": "sft",
        "resolved": {"tokenizer": ChatTokenizer(base, version).metadata()},
    }
    config = {"directory": "prepared", "chat_format_version": version}
    assert load_checkpoint_tokenizer(config, metadata).format_version == version
    assert config["chat_format_version"] == version
    assert load_checkpoint_tokenizer({"directory": "prepared"}, metadata).format_version == version
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        load_checkpoint_tokenizer({**config, "chat_format_version": 3 - version}, metadata)
    with pytest.raises(ValueError, match="require an SFT"):
        load_checkpoint_tokenizer(config, {"training_phase": "base"})


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
        arguments(["hello", "--experiment", "experiment", option, value])
    assert error.value.code == 2
