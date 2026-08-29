"""Generate reproducible procedural pre-pretraining corpora for SpeckGym."""

import hashlib
import json
import os
import random
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

from speck.common import base_dir
from speck.dataset import TokenShardWriter, derive_source_quotas

FORMAT = "speck_procedural_tokens"
FORMAT_VERSION = 1
FAMILIES = ("hierarchy", "binding", "state", "set_union", "composition")
RUN_DATASETS = {
    "B": "B-RandomSymbols",
    "C": "C-ShuffledGym",
    "D": "D-FormalStructure",
    "E": "E-SpeckGym",
}

_START = 0
_QUERY = 1
_ANSWER = 2
_SEP = 3
_LINK = 4
_UPDATE = 5
_UNION = 6
_OPEN = 7
_CLOSE = 8
_FAMILY_TOKEN = {family: 9 + index for index, family in enumerate(FAMILIES)}
_VALUE_START = 16


def _json_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed(seed, *parts):
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_speckgym_config(experiment="experiments/SpeckGym-v0"):
    """Load and validate the checked SpeckGym suite contract."""

    experiment = Path(experiment)
    path = experiment / "gym.json" if experiment.is_dir() else experiment
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("format_version") != 1:
        raise ValueError("unsupported SpeckGym configuration")
    required = {
        "base_experiment",
        "batch_tokens",
        "total_requested_tokens",
        "checkpoint_tokens",
        "evaluation",
        "procedural",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"SpeckGym configuration is missing: {', '.join(sorted(missing))}")
    procedural = config["procedural"]
    for key in (
        "seed",
        "updates",
        "sequence_length",
        "symbol_count",
        "validation_sequences",
        "reserve_sequences",
        "shard_tokens",
        "formal",
    ):
        if key not in procedural:
            raise ValueError(f"SpeckGym procedural configuration is missing {key}")
    if procedural["symbol_count"] < _VALUE_START + 2:
        raise ValueError("SpeckGym requires enough symbols for protocol and values")
    if config["batch_tokens"] % procedural["sequence_length"]:
        raise ValueError("SpeckGym batch tokens must align with procedural sequences")
    warmup_tokens = config["batch_tokens"] * procedural["updates"]
    if warmup_tokens >= config["total_requested_tokens"]:
        raise ValueError("SpeckGym warm-up must be shorter than the total token budget")
    config["experiment"] = str(path.parent.resolve())
    config["base_experiment"] = str((path.parent / config["base_experiment"]).resolve())
    config["evaluation"]["standard_config"] = str(
        (path.parent / config["evaluation"]["standard_config"]).resolve()
    )
    config["warmup_tokens"] = warmup_tokens
    return config


def resolve_training_phase(config, base_configs, run, phase, cache_dir=None):
    """Resolve one A-E phase into the ordinary base-training configuration contract."""

    if run not in {"A", *RUN_DATASETS}:
        raise ValueError("SpeckGym run must be one of A, B, C, D, or E")
    if phase not in {"warmup", "language"}:
        raise ValueError("SpeckGym phase must be warmup or language")
    if run == "A" and phase == "warmup":
        raise ValueError("baseline A has no procedural warm-up")
    configs = deepcopy(base_configs)
    train = configs["train"]
    data = configs["data"]
    cache_dir = Path(cache_dir or base_dir()).expanduser().resolve()
    warmup_tokens = config["warmup_tokens"]
    group = f"SpeckGym-v0-{run}"
    if phase == "warmup":
        name = f"{group}-warmup"
        data["output_dir"] = str(
            cache_dir
            / "data"
            / config["procedural"].get("output_name", "SpeckGym-v0")
            / RUN_DATASETS[run]
        )
        train.update(
            checkpoint_tokens=[],
            eval_every=100,
            eval_tokens=262_144,
            final_eval_tokens=1_048_576,
            global_token_offset=0,
            initialization=None,
            run=name,
            save_every=0,
            train_tokens=warmup_tokens,
            training_phase="procedural_warmup",
            wandb_group=group,
        )
    else:
        name = group
        initialized = run != "A"
        train.update(
            checkpoint_tokens=config["checkpoint_tokens"],
            global_token_offset=warmup_tokens if initialized else 0,
            initialization=(
                {
                    "kind": "backbone_checkpoint",
                    "checkpoint_dir": str(cache_dir / "checkpoints" / f"{group}-warmup"),
                    "step": config["procedural"]["updates"],
                }
                if initialized
                else None
            ),
            run=name,
            save_every=0,
            train_tokens=(
                config["total_requested_tokens"] - warmup_tokens
                if initialized
                else config["total_requested_tokens"]
            ),
            training_phase="language",
            wandb_group=group,
        )
    train["batch_tokens"] = config["batch_tokens"]
    train["output_dir"] = str(cache_dir / "checkpoints" / name)
    return configs


def symbol_ids(vocab_size, count, seed):
    """Choose a stable abstract-symbol subset of the full model vocabulary."""

    if not isinstance(vocab_size, int) or vocab_size <= count + 3:
        raise ValueError("model vocabulary is too small for the procedural symbols")
    rng = random.Random(_seed(seed, "surface-symbols"))
    return tuple(rng.sample(range(3, vocab_size), count))


def generate_shuffle_dyck(k, length, p_open, max_depth, rng):
    """Generate Hu et al.'s k-Shuffle Dyck sequence, truncated at ``length``."""

    _positive_integer(k, "k")
    _positive_integer(length, "length")
    _positive_integer(max_depth, "max_depth")
    if not isinstance(p_open, (int, float)) or not 0 <= p_open <= 1:
        raise ValueError("p_open must be between zero and one")
    counts = [0] * k
    sequence = []
    while len(sequence) < length:
        depth = sum(counts)
        if depth == 0:
            bracket = rng.randrange(k)
            sequence.append(bracket)
            counts[bracket] += 1
        elif depth >= max_depth:
            bracket = rng.choice([index for index, count in enumerate(counts) if count])
            sequence.append(bracket + k)
            counts[bracket] -= 1
        elif rng.random() < p_open:
            bracket = rng.randrange(k)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            bracket = rng.choice([index for index, count in enumerate(counts) if count])
            sequence.append(bracket + k)
            counts[bracket] -= 1
    return sequence


def _values(rng, count, *, exclude=()):
    available = [value for value in range(_VALUE_START, 128) if value not in exclude]
    if count > len(available):
        raise ValueError("procedural example exhausted the abstract symbol pool")
    return rng.sample(available, count)


def _hierarchy_episode(rng, stage):
    depth = rng.randint(2, (4, 8, 16)[stage])
    labels = _values(rng, depth)
    payload = [_START, _FAMILY_TOKEN["hierarchy"]]
    for label in labels:
        payload.extend((_OPEN, label))
    for label in reversed(labels):
        payload.extend((_CLOSE, label))
    child_index = rng.randrange(1, len(labels))
    payload.extend((_QUERY, labels[child_index], _ANSWER, labels[child_index - 1], _SEP))
    return payload


def _binding_episode(rng, stage):
    pairs = (4, 8, 16)[stage]
    labels = _values(rng, pairs * 2)
    keys, values = labels[:pairs], labels[pairs:]
    order = list(range(pairs))
    rng.shuffle(order)
    payload = [_START, _FAMILY_TOKEN["binding"]]
    for index in order:
        payload.extend((keys[index], _LINK, values[index], _SEP))
    query = rng.randrange(pairs)
    payload.extend((_QUERY, keys[query], _ANSWER, values[query], _SEP))
    return payload


def _state_episode(rng, stage):
    register_count = (3, 5, 8)[stage]
    update_count = (4, 12, 24)[stage]
    labels = _values(rng, register_count * 2 + update_count)
    registers = labels[:register_count]
    initial = labels[register_count : register_count * 2]
    updates = labels[register_count * 2 :]
    state = dict(zip(registers, initial))
    payload = [_START, _FAMILY_TOKEN["state"]]
    for register in registers:
        payload.extend((register, _LINK, state[register], _SEP))
    for value in updates:
        register = rng.choice(registers)
        state[register] = value
        payload.extend((_UPDATE, register, _LINK, value, _SEP))
    query = rng.choice(registers)
    payload.extend((_QUERY, query, _ANSWER, state[query], _SEP))
    return payload


def _set_episode(rng, stage):
    maximum = (3, 6, 10)[stage]
    left_size = rng.randint(1, maximum)
    right_size = rng.randint(1, maximum)
    labels = _values(rng, 2 + left_size + right_size)
    left_name, right_name = labels[:2]
    left = labels[2 : 2 + left_size]
    right = labels[2 + left_size :]
    answer = list(dict.fromkeys((*left, *right)))
    rng.shuffle(answer)
    return [
        _START,
        _FAMILY_TOKEN["set_union"],
        left_name,
        _OPEN,
        *left,
        _CLOSE,
        _SEP,
        right_name,
        _OPEN,
        *right,
        _CLOSE,
        _SEP,
        _QUERY,
        left_name,
        _UNION,
        right_name,
        _ANSWER,
        *answer,
        _SEP,
    ]


def _composition_episode(rng, stage):
    hops = (2, 4, 6)[stage]
    labels = _values(rng, hops * 2 + 1)
    functions = labels[:hops]
    values = labels[hops : hops * 2 + 1]
    rules = [(functions[index], values[index], values[index + 1]) for index in range(hops)]
    rng.shuffle(rules)
    payload = [_START, _FAMILY_TOKEN["composition"]]
    for function, source, target in rules:
        payload.extend((function, source, _LINK, target, _SEP))
    payload.extend((_QUERY, values[0], *functions, _ANSWER, values[-1], _SEP))
    return payload


_EPISODES = {
    "hierarchy": _hierarchy_episode,
    "binding": _binding_episode,
    "state": _state_episode,
    "set_union": _set_episode,
    "composition": _composition_episode,
}


def generate_gym_block(family, length, seed, split, block_index, total_train_blocks):
    """Generate one deterministic SpeckGym block in its logical 128-symbol alphabet."""

    if family not in _EPISODES:
        raise ValueError(f"unknown SpeckGym family: {family}")
    _positive_integer(length, "length")
    rng = random.Random(_seed(seed, "gym", family, split, block_index))
    if split == "train":
        stage = min(2, block_index * 3 // max(1, total_train_blocks))
    else:
        stage = 2
    block = []
    while len(block) < length:
        block.extend(_EPISODES[family](rng, stage))
    return block[:length]


def _mapped(block, surface):
    return [surface[token] for token in block]


def _source_manifest(source_id, generator, requested, reserve, split_shards):
    splits = {}
    for split, shards in split_shards.items():
        tokens = sum(shard["tokens"] for shard in shards)
        splits[split] = {
            "requested_tokens": requested if split == "train" else tokens,
            "reserve_tokens": reserve if split == "train" else 0,
            "tokens": tokens,
            "shards": [
                {**shard, "path": f"sources/{source_id}/{shard['path']}"} for shard in shards
            ],
        }
    return {"id": source_id, "generator": generator, "splits": splits}


def _write_dataset(
    output_dir,
    *,
    tokenizer,
    requested_train_tokens,
    sequence_length,
    shard_tokens,
    seed,
    surface,
    source_generators,
    mixture,
    validation_sequences,
    reserve_sequences,
    restart,
):
    output_dir = Path(output_dir)
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise FileExistsError(f"procedural dataset already exists: {output_dir}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete procedural build exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sources = [{"id": source_id} for source_id in source_generators]
        quotas, phases = derive_source_quotas(sources, mixture, requested_train_tokens)
        reserve_tokens = reserve_sequences * sequence_length
        source_manifests = []
        for source_id, generate in source_generators.items():
            if quotas[source_id] % sequence_length:
                raise ValueError(f"procedural quota for {source_id} is not sequence-aligned")
            directory = staging / "sources" / source_id
            split_shards = {}
            split_blocks = {
                "train": quotas[source_id] // sequence_length + reserve_sequences,
                "val": validation_sequences,
            }
            for split, blocks in split_blocks.items():
                writer = TokenShardWriter(directory, split, shard_tokens)
                for block_index in range(blocks):
                    block = generate(split, block_index, quotas[source_id] // sequence_length)
                    if len(block) != sequence_length:
                        raise ValueError("procedural generator returned a mis-sized block")
                    if min(block) < 0 or max(block) >= tokenizer.vocab_size:
                        raise ValueError("procedural generator emitted an out-of-vocabulary token")
                    writer.write(block)
                split_shards[split] = writer.finish()
            source_manifests.append(
                _source_manifest(
                    source_id,
                    generate.provenance,
                    quotas[source_id],
                    reserve_tokens,
                    split_shards,
                )
            )
        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "dtype": "<u2",
            "requested_train_tokens": requested_train_tokens,
            "mixture": {"phases": phases, "source_quotas": quotas},
            "preparation": {
                "seed": seed,
                "sequence_length": sequence_length,
                "shard_tokens": shard_tokens,
                "validation_sequences_per_source": validation_sequences,
                "reserve_sequences_per_source": reserve_sequences,
            },
            "tokenizer": {
                "fingerprint": tokenizer.fingerprint(),
                "vocab_size": tokenizer.vocab_size,
                "bos_token_id": tokenizer.bos_id,
                "eos_token_id": tokenizer.eos_id,
            },
            "encoding": {
                "kind": "direct_abstract_symbols",
                "surface_token_ids": list(surface),
                "fingerprint": _json_fingerprint(list(surface)),
            },
            "sources": source_manifests,
            "splits": {
                split: {
                    "tokens": sum(source["splits"][split]["tokens"] for source in source_manifests)
                }
                for split in ("train", "val")
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _generator(function, provenance):
    function.provenance = provenance
    return function


def prepare_speckgym(config, tokenizer, output_dir=None, restart=False):
    """Prepare B-E procedural corpora from one validated suite configuration."""

    procedural = config["procedural"]
    seed = procedural["seed"]
    sequence_length = procedural["sequence_length"]
    requested = config["batch_tokens"] * procedural["updates"]
    surface = symbol_ids(tokenizer.vocab_size, procedural["symbol_count"], seed)
    if len(surface) != 128:
        raise ValueError("SpeckGym v0 uses exactly 128 abstract symbols")
    final_output_dir = Path(
        output_dir or Path(base_dir()) / "data" / procedural.get("output_name", "SpeckGym-v0")
    ).expanduser()
    if final_output_dir.exists():
        raise FileExistsError(f"SpeckGym datasets already exist: {final_output_dir}")
    output_dir = final_output_dir.with_name(final_output_dir.name + ".building")
    if output_dir.exists():
        if not restart:
            raise FileExistsError(f"incomplete SpeckGym build exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    common = {
        "tokenizer": tokenizer,
        "requested_train_tokens": requested,
        "sequence_length": sequence_length,
        "shard_tokens": procedural["shard_tokens"],
        "seed": seed,
        "surface": surface,
        "validation_sequences": procedural["validation_sequences"],
        "reserve_sequences": procedural["reserve_sequences"],
        "restart": False,
    }
    equal_weights = {family: 100 // len(FAMILIES) for family in FAMILIES}
    gym_mixture = {"phases": [{"end_tokens": requested, "weights": equal_weights}]}

    def gym_generator(family):
        def generate(split, block_index, train_blocks):
            logical = generate_gym_block(
                family, sequence_length, seed, split, block_index, train_blocks
            )
            return _mapped(logical, surface)

        return _generator(
            generate,
            {"name": "speckgym", "version": 0, "family": family, "seed": seed},
        )

    gym_generators = {family: gym_generator(family) for family in FAMILIES}
    manifests = {
        "E": _write_dataset(
            output_dir / "E-SpeckGym",
            source_generators=gym_generators,
            mixture=gym_mixture,
            **common,
        )
    }

    def shuffled_generator(family):
        def generate(split, block_index, train_blocks):
            logical = generate_gym_block(
                family, sequence_length, seed, split, block_index, train_blocks
            )
            random.Random(_seed(seed, "shuffle", family, split, block_index)).shuffle(logical)
            return _mapped(logical, surface)

        return _generator(
            generate,
            {
                "name": "shuffled_speckgym",
                "version": 0,
                "family": family,
                "seed": seed,
                "source": manifests["E"]["encoding"]["fingerprint"],
            },
        )

    manifests["C"] = _write_dataset(
        output_dir / "C-ShuffledGym",
        source_generators={family: shuffled_generator(family) for family in FAMILIES},
        mixture=gym_mixture,
        **common,
    )

    histogram = Counter()
    family_blocks = requested // len(FAMILIES) // sequence_length
    for family in FAMILIES:
        for block_index in range(family_blocks):
            histogram.update(
                generate_gym_block(
                    family, sequence_length, seed, "train", block_index, family_blocks
                )
            )
    population = tuple(range(len(surface)))
    weights = tuple(histogram[index] for index in population)

    def random_generate(split, block_index, train_blocks):
        del train_blocks
        rng = random.Random(_seed(seed, "random", split, block_index))
        return _mapped(rng.choices(population, weights=weights, k=sequence_length), surface)

    random_generate = _generator(
        random_generate,
        {
            "name": "iid_symbols",
            "version": 0,
            "seed": seed,
            "reference_unigram_sha256": _json_fingerprint(dict(sorted(histogram.items()))),
        },
    )
    single_mixture = {"phases": [{"end_tokens": requested, "weights": {"symbols": 100}}]}
    manifests["B"] = _write_dataset(
        output_dir / "B-RandomSymbols",
        source_generators={"symbols": random_generate},
        mixture=single_mixture,
        **common,
    )

    formal = procedural["formal"]

    def formal_generate(split, block_index, train_blocks):
        del train_blocks
        rng = random.Random(_seed(seed, "shuffle-dyck", split, block_index))
        logical = generate_shuffle_dyck(
            formal["k"],
            sequence_length,
            formal["p_open"],
            formal["max_depth"],
            rng,
        )
        return _mapped(logical, surface)

    formal_generate = _generator(
        formal_generate,
        {
            "name": "k_shuffle_dyck",
            "version": 1,
            "seed": seed,
            **formal,
        },
    )
    manifests["D"] = _write_dataset(
        output_dir / "D-FormalStructure",
        source_generators={"shuffle_dyck": formal_generate},
        mixture={"phases": [{"end_tokens": requested, "weights": {"shuffle_dyck": 100}}]},
        **common,
    )
    os.replace(output_dir, final_output_dir)
    return manifests
