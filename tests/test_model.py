import torch

from speck.model import Config, Llama, MLP, RMSNorm


def tiny_model():
    config = Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    model = Llama(config)
    model.init_weights()
    return model


def test_default_model_is_exactly_50m():
    with torch.device("meta"):
        model = Llama()
    assert model.parameter_count() == 50_055_552
    assert model.config.export()["architectures"] == ["LlamaForCausalLM"]


def test_llama_structure_and_tied_embeddings():
    model = tiny_model()
    layer = model.model.layers[0]
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.mlp, MLP)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_forward_loss_and_optimizer():
    model = tiny_model()
    tokens = torch.randint(0, 32, (2, 8))
    assert model(tokens).shape == (2, 8, 32)
    loss = model(tokens, tokens)
    loss.backward()
    optimizer = model.optimizer()
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
    optimizer.step()


def test_cached_decode_matches_full_forward():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (1, 7))
    expected = model(tokens)[:, -1]
    cache = model.cache(length=16)
    model(tokens[:, :6], cache=cache)
    actual = model(tokens[:, 6:], cache=cache)[:, -1]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
