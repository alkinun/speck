"""candidate mutation, repair, and static architecture metrics."""

import hashlib
import json
import random
from dataclasses import dataclass, replace

from speck.model import Config, LayerConfig


mutation_operators = (
    "add_layer",
    "remove_layer",
    "change_hidden_size",
    "change_ffn_width",
    "toggle_attention",
    "change_kv_heads",
    "alter_attention_placement",
)


@dataclass(frozen=True)
class SearchSpace:
    min_layers: int
    max_layers: int
    hidden_size_min: int
    hidden_size_max: int
    hidden_size_step: int
    intermediate_size_min: int
    intermediate_size_max: int
    intermediate_size_step: int
    kv_heads: tuple[int, ...]
    min_attention_layers: int = 1
    max_attention_layers: int | None = None
    min_parameters: int | None = None
    max_parameters: int | None = None
    max_kv_bytes_per_token: int | None = None
    cache_dtype_bytes: int = 2

    def __post_init__(self):
        if self.min_layers < 1 or self.max_layers < self.min_layers:
            raise ValueError("invalid layer range")
        for minimum, maximum, step in (
            (self.hidden_size_min, self.hidden_size_max, self.hidden_size_step),
            (
                self.intermediate_size_min,
                self.intermediate_size_max,
                self.intermediate_size_step,
            ),
        ):
            if minimum < 1 or maximum < minimum or step < 1:
                raise ValueError("invalid dimension range")
        kv_heads = tuple(sorted(set(self.kv_heads)))
        object.__setattr__(self, "kv_heads", kv_heads)
        if not kv_heads or kv_heads[0] < 1:
            raise ValueError("kv head choices must be positive")
        maximum_attention = (
            self.max_attention_layers
            if self.max_attention_layers is not None
            else self.max_layers
        )
        object.__setattr__(self, "max_attention_layers", maximum_attention)
        if not 0 <= self.min_attention_layers <= self.min_layers:
            raise ValueError("minimum attention layers exceed minimum depth")
        if maximum_attention < self.min_attention_layers:
            raise ValueError("invalid attention layer range")
        if self.cache_dtype_bytes < 1:
            raise ValueError("cache dtype bytes must be positive")
        if (
            self.min_parameters is not None
            and self.max_parameters is not None
            and self.min_parameters > self.max_parameters
        ):
            raise ValueError("invalid parameter range")

    @classmethod
    def from_dict(cls, settings):
        values = dict(settings)
        values["kv_heads"] = tuple(values["kv_heads"])
        return cls(**values)


@dataclass(frozen=True)
class MutationResult:
    config: Config
    mutation: dict
    repairs: tuple[dict, ...]


def canonical_settings(config):
    return config.settings()


def architecture_hash(config):
    payload = json.dumps(
        canonical_settings(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def parameter_count(config):
    layers = config.layers
    embedding_size = config.embedding_size
    total = config.vocab_size * embedding_size
    input_size = embedding_size
    for layer in layers:
        if input_size != layer.hidden_size:
            total += input_size * layer.hidden_size
        if layer.num_key_value_heads is not None:
            kv_size = layer.num_key_value_heads * config.head_dim
            total += 2 * layer.hidden_size * layer.hidden_size
            total += 2 * layer.hidden_size * kv_size
            total += 2 * config.head_dim
            total += layer.hidden_size
        total += 3 * layer.hidden_size * layer.intermediate_size
        total += layer.hidden_size
        input_size = layer.hidden_size
    total += layers[-1].hidden_size
    if layers[-1].hidden_size != embedding_size:
        total += layers[-1].hidden_size * embedding_size
    return total


def kv_bytes_per_token(config, dtype_bytes=2):
    return 2 * dtype_bytes * config.head_dim * sum(
        layer.num_key_value_heads or 0 for layer in config.layers
    )


def _grid(minimum, maximum, step):
    return tuple(range(minimum, maximum + 1, step))


def _nearest(value, choices):
    return min(choices, key=lambda choice: (abs(choice - value), choice))


def _valid_kv_heads(layer, config, space):
    query_heads = layer.hidden_size // config.head_dim
    return tuple(heads for heads in space.kv_heads if query_heads % heads == 0)


def _with_layers(config, layers):
    return replace(config, layers=tuple(layers))


def repair(config, space):
    layers = list(config.layers)
    repairs = []
    while len(layers) < space.min_layers:
        layers.append(layers[-1])
        repairs.append({"kind": "add_layer", "index": len(layers) - 1})
    if len(layers) > space.max_layers:
        removed = len(layers) - space.max_layers
        del layers[space.max_layers:]
        repairs.append({"kind": "remove_layers", "count": removed})

    hidden_sizes = tuple(
        size
        for size in _grid(
            space.hidden_size_min, space.hidden_size_max, space.hidden_size_step
        )
        if size % config.head_dim == 0
    )
    if not hidden_sizes:
        raise ValueError("hidden size range has no multiple of the head dimension")
    intermediate_sizes = _grid(
        space.intermediate_size_min,
        space.intermediate_size_max,
        space.intermediate_size_step,
    )
    for index, layer in enumerate(layers):
        hidden_size = _nearest(layer.hidden_size, hidden_sizes)
        intermediate_size = _nearest(layer.intermediate_size, intermediate_sizes)
        changed = {}
        if hidden_size != layer.hidden_size:
            changed["hidden_size"] = {"from": layer.hidden_size, "to": hidden_size}
        if intermediate_size != layer.intermediate_size:
            changed["intermediate_size"] = {
                "from": layer.intermediate_size,
                "to": intermediate_size,
            }
        updated = replace(
            layer, hidden_size=hidden_size, intermediate_size=intermediate_size
        )
        if updated.num_key_value_heads is not None:
            choices = _valid_kv_heads(updated, config, space)
            if not choices:
                raise ValueError("no kv head choice divides the repaired query heads")
            kv_heads = _nearest(updated.num_key_value_heads, choices)
            if kv_heads != updated.num_key_value_heads:
                changed["num_key_value_heads"] = {
                    "from": updated.num_key_value_heads,
                    "to": kv_heads,
                }
                updated = replace(updated, num_key_value_heads=kv_heads)
        layers[index] = updated
        if changed:
            repairs.append({"kind": "repair_layer", "index": index, **changed})

    attention = [
        index for index, layer in enumerate(layers) if layer.num_key_value_heads is not None
    ]
    if len(attention) < space.min_attention_layers:
        for index, layer in enumerate(layers):
            if layer.num_key_value_heads is None:
                choices = _valid_kv_heads(layer, config, space)
                if not choices:
                    continue
                layers[index] = replace(layer, num_key_value_heads=choices[0])
                repairs.append(
                    {"kind": "enable_attention", "index": index, "kv_heads": choices[0]}
                )
                attention.append(index)
                if len(attention) == space.min_attention_layers:
                    break
    if len(attention) > space.max_attention_layers:
        for index in reversed(attention[space.max_attention_layers:]):
            layers[index] = replace(layers[index], num_key_value_heads=None)
            repairs.append({"kind": "disable_attention", "index": index})

    attention_count = sum(
        layer.num_key_value_heads is not None for layer in layers
    )
    if not space.min_attention_layers <= attention_count <= space.max_attention_layers:
        raise ValueError("candidate cannot satisfy the attention layer range")

    repaired = _with_layers(config, layers)
    parameters = parameter_count(repaired)
    cache_bytes = kv_bytes_per_token(repaired, space.cache_dtype_bytes)
    if space.min_parameters is not None and parameters < space.min_parameters:
        raise ValueError("candidate is below the minimum parameter count")
    if space.max_parameters is not None and parameters > space.max_parameters:
        raise ValueError("candidate exceeds the maximum parameter count")
    if (
        space.max_kv_bytes_per_token is not None
        and cache_bytes > space.max_kv_bytes_per_token
    ):
        raise ValueError("candidate exceeds the kv cache limit")
    return repaired, tuple(repairs)


def _neighbor(value, choices, rng):
    index = choices.index(value)
    neighbors = []
    if index:
        neighbors.append(choices[index - 1])
    if index + 1 < len(choices):
        neighbors.append(choices[index + 1])
    if not neighbors:
        raise ValueError("dimension has no neighboring choice")
    return rng.choice(neighbors)


def _available_operators(config, space):
    layers = config.layers
    attention = [layer.num_key_value_heads is not None for layer in layers]
    hidden_choices = tuple(
        size
        for size in _grid(
            space.hidden_size_min, space.hidden_size_max, space.hidden_size_step
        )
        if size % config.head_dim == 0
    )
    intermediate_choices = _grid(
        space.intermediate_size_min,
        space.intermediate_size_max,
        space.intermediate_size_step,
    )
    available = []
    if len(layers) < space.max_layers:
        available.append("add_layer")
    if len(layers) > space.min_layers:
        available.append("remove_layer")
    if len(hidden_choices) > 1:
        available.append("change_hidden_size")
    if len(intermediate_choices) > 1:
        available.append("change_ffn_width")
    if any(attention) or sum(attention) < space.max_attention_layers:
        available.append("toggle_attention")
    if any(
        layer.num_key_value_heads is not None
        and len(_valid_kv_heads(layer, config, space)) > 1
        for layer in layers
    ):
        available.append("change_kv_heads")
    if any(attention) and not all(attention):
        available.append("alter_attention_placement")
    return tuple(available)


def available_mutations(config, space):
    return _available_operators(config, space)


def mutate(config, space, seed, operator=None):
    rng = random.Random(seed)
    available = _available_operators(config, space)
    if operator is None:
        operator = rng.choice(available)
    if operator not in available:
        raise ValueError(f"mutation is not available: {operator}")
    layers = list(config.layers)
    mutation = {"operator": operator, "seed": seed}
    mutation_repairs = []

    if operator == "add_layer":
        index = rng.randrange(len(layers) + 1)
        source = min(index, len(layers) - 1)
        layers.insert(index, layers[source])
        mutation.update(index=index, source=source)
    elif operator == "remove_layer":
        index = rng.randrange(len(layers))
        removed = layers.pop(index)
        mutation.update(index=index, removed=removed.__dict__)
    elif operator == "change_hidden_size":
        index = rng.randrange(len(layers))
        choices = tuple(
            size
            for size in _grid(
                space.hidden_size_min, space.hidden_size_max, space.hidden_size_step
            )
            if size % config.head_dim == 0
        )
        value = _neighbor(layers[index].hidden_size, choices, rng)
        mutation.update(index=index, old=layers[index].hidden_size, new=value)
        updated = replace(layers[index], hidden_size=value)
        if updated.num_key_value_heads is not None:
            kv_choices = _valid_kv_heads(updated, config, space)
            if not kv_choices:
                raise ValueError("no valid kv heads for the mutated hidden size")
            kv_heads = _nearest(updated.num_key_value_heads, kv_choices)
            if kv_heads != updated.num_key_value_heads:
                mutation_repairs.append({
                    "kind": "repair_layer",
                    "index": index,
                    "num_key_value_heads": {
                        "from": updated.num_key_value_heads,
                        "to": kv_heads,
                    },
                })
                updated = replace(updated, num_key_value_heads=kv_heads)
        layers[index] = updated
    elif operator == "change_ffn_width":
        index = rng.randrange(len(layers))
        choices = _grid(
            space.intermediate_size_min,
            space.intermediate_size_max,
            space.intermediate_size_step,
        )
        value = _neighbor(layers[index].intermediate_size, choices, rng)
        mutation.update(index=index, old=layers[index].intermediate_size, new=value)
        layers[index] = replace(layers[index], intermediate_size=value)
    elif operator == "toggle_attention":
        index = rng.randrange(len(layers))
        old = layers[index].num_key_value_heads
        if old is None:
            choices = _valid_kv_heads(layers[index], config, space)
            if not choices:
                raise ValueError("no valid kv heads for the selected layer")
            new = rng.choice(choices)
        else:
            new = None
        mutation.update(index=index, old=old, new=new)
        layers[index] = replace(layers[index], num_key_value_heads=new)
    elif operator == "change_kv_heads":
        indices = [
            index
            for index, layer in enumerate(layers)
            if layer.num_key_value_heads is not None
            and len(_valid_kv_heads(layer, config, space)) > 1
        ]
        if not indices:
            raise ValueError("no attention layer has another valid kv head count")
        index = rng.choice(indices)
        choices = _valid_kv_heads(layers[index], config, space)
        old = layers[index].num_key_value_heads
        new = _neighbor(old, choices, rng)
        mutation.update(index=index, old=old, new=new)
        layers[index] = replace(layers[index], num_key_value_heads=new)
    else:
        sources = [
            index for index, layer in enumerate(layers) if layer.num_key_value_heads is not None
        ]
        targets = [
            index for index, layer in enumerate(layers) if layer.num_key_value_heads is None
        ]
        source = rng.choice(sources)
        target = rng.choice(targets)
        kv_heads = layers[source].num_key_value_heads
        target_choices = _valid_kv_heads(layers[target], config, space)
        if not target_choices:
            raise ValueError("no valid kv heads for the attention target")
        repaired_kv_heads = _nearest(kv_heads, target_choices)
        layers[source] = replace(layers[source], num_key_value_heads=None)
        layers[target] = replace(
            layers[target], num_key_value_heads=repaired_kv_heads
        )
        mutation.update(source=source, target=target, kv_heads=kv_heads)
        if repaired_kv_heads != kv_heads:
            mutation_repairs.append({
                "kind": "repair_layer",
                "index": target,
                "num_key_value_heads": {
                    "from": kv_heads,
                    "to": repaired_kv_heads,
                },
            })

    repaired, repairs = repair(_with_layers(config, layers), space)
    return MutationResult(repaired, mutation, tuple(mutation_repairs) + repairs)


def crossover(left, right, space, seed):
    left_settings = left.settings()
    right_settings = right.settings()
    left_layers = left_settings.pop("layers")
    right_layers = right_settings.pop("layers")
    if left_settings != right_settings:
        raise ValueError("crossover parents must share global model settings")
    if len(left.layers) < 2 or len(right.layers) < 2:
        raise ValueError("crossover parents must contain at least two layers")
    rng = random.Random(seed)
    left_cut = rng.randrange(1, len(left.layers))
    ratio = left_cut / len(left.layers)
    right_cut = min(
        len(right.layers) - 1,
        max(1, round(ratio * len(right.layers))),
    )
    layers = left.layers[:left_cut] + right.layers[right_cut:]
    repaired, repairs = repair(_with_layers(left, layers), space)
    operation = {
        "operator": "crossover",
        "seed": seed,
        "left_hash": architecture_hash(left),
        "right_hash": architecture_hash(right),
        "left_cut": left_cut,
        "right_cut": right_cut,
    }
    return MutationResult(repaired, operation, repairs)


def architecture_distance(left, right, space):
    hidden_span = max(1, space.hidden_size_max - space.hidden_size_min)
    intermediate_span = max(
        1, space.intermediate_size_max - space.intermediate_size_min
    )
    kv_span = max(space.kv_heads)
    distance = 0.0
    for index in range(space.max_layers):
        left_layer = left.layers[index] if index < len(left.layers) else None
        right_layer = right.layers[index] if index < len(right.layers) else None
        if left_layer is None or right_layer is None:
            distance += float(left_layer is not right_layer)
            continue
        distance += abs(left_layer.hidden_size - right_layer.hidden_size) / hidden_span
        distance += (
            abs(left_layer.intermediate_size - right_layer.intermediate_size)
            / intermediate_span
        )
        left_attention = left_layer.num_key_value_heads is not None
        right_attention = right_layer.num_key_value_heads is not None
        distance += float(left_attention != right_attention)
        distance += abs(
            (left_layer.num_key_value_heads or 0)
            - (right_layer.num_key_value_heads or 0)
        ) / kv_span
    return distance / (4 * space.max_layers)


def novelty(config, population, space, neighbors=3):
    distances = sorted(
        architecture_distance(config, other, space)
        for other in population
        if architecture_hash(config) != architecture_hash(other)
    )
    if not distances:
        return 1.0
    selected = distances[:neighbors]
    return sum(selected) / len(selected)
