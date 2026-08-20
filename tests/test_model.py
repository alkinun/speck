import torch

from speck.model import Attention, Config, LayerConfig, MLP, RMSNorm, SpeckForCausalLM


def tiny_model():
    config = Config(
        vocab_size=32,
        layers=(
            LayerConfig(hidden_size=16, intermediate_size=32, num_key_value_heads=2),
            LayerConfig(hidden_size=16, intermediate_size=32, num_key_value_heads=None),
        ),
        head_dim=4,
        max_position_embeddings=32,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    return model


def test_default_model_is_exactly_200m():
    with torch.device("meta"):
        model = SpeckForCausalLM()
    assert model.parameter_count() == 199_511_808
    assert model.config.export()["architectures"] == ["SpeckForCausalLM"]


def test_sparse_attention_structure_and_tied_embeddings():
    model = tiny_model()
    attention_layer, mlp_layer = model.model.layers
    assert isinstance(attention_layer.attention_norm, RMSNorm)
    assert isinstance(attention_layer.self_attn, Attention)
    assert attention_layer.self_attn.q_norm.weight.shape == (4,)
    assert mlp_layer.self_attn is None
    assert isinstance(mlp_layer.mlp, MLP)
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


def test_muon_optimizer_covers_every_parameter():
    model = tiny_model()
    optimizer = model.optimizer(name="muon")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_cached_decode_matches_full_forward():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (1, 7))
    expected = model(tokens)[:, -1]
    cache = model.cache(length=16)
    model(tokens[:, :6], cache=cache)
    actual = model(tokens[:, 6:], cache=cache)[:, -1]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_heterogeneous_layers_project_and_cache():
    config = Config(
        vocab_size=32,
        layers=(
            LayerConfig(hidden_size=16, intermediate_size=32, num_key_value_heads=2),
            LayerConfig(hidden_size=8, intermediate_size=24, num_key_value_heads=1),
        ),
        head_dim=4,
        max_position_embeddings=16,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    input_projection = model.model.layers[1].input_projection
    output_projection = model.model.output_projection
    assert isinstance(input_projection, torch.nn.Linear)
    assert isinstance(output_projection, torch.nn.Linear)
    assert input_projection.weight.shape == (8, 16)
    assert output_projection.weight.shape == (16, 8)
    cache = model.cache(length=8)
    assert cache.keys[0].shape == (1, 2, 8, 4)
    assert cache.keys[1].shape == (1, 1, 8, 4)
    assert cache.bytes_per_token() == 96
    tokens = torch.randint(0, 32, (1, 4))
    assert model(tokens, cache=cache).shape == (1, 4, 32)


def test_legacy_config_expands_to_layers():
    config = Config.from_dict({
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "attention_every": 2,
        "expected_parameters": 123,
    })
    assert [layer.num_key_value_heads for layer in config.layers] == [2, None, 2]


def test_exported_config_round_trips():
    config = Config(rope_theta=500000.0)
    assert Config.from_dict(config.export()).settings() == config.settings()
