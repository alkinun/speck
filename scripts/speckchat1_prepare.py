#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==5.0.1",
# ]
# ///
"""Build and publish the 300K-example SpeckChat1 instruction dataset."""

from dataclasses import dataclass

OUTPUT_REPO = "specklabs/SpeckChat1"
SEED = 42
TOTAL_SAMPLES = 300_000
OPENHERMES_SAMPLES = 180_000


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    revision: str


OPENHERMES = SourceSpec(
    "teknium/OpenHermes-2.5",
    "b82037821055c377bed0d495e72e46de3bc72e84",
)
MAGPIE = SourceSpec(
    "Magpie-Align/Magpie-Air-MT-300K-v0.1",
    "aa5bce5ce1fec181c6143c545b936cbd43a9e922",
)
NO_ROBOTS = SourceSpec(
    "HuggingFaceH4/no_robots",
    "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
)
COCONOT = SourceSpec(
    "allenai/coconot",
    "2cbe16aabf9069f17e48c8daad8aeabc29469eb7",
)
SOURCES = (OPENHERMES, MAGPIE, NO_ROBOTS, COCONOT)
ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def convert_sharegpt(example, source):
    messages = [
        {"role": ROLE_MAP[turn["from"]], "content": turn["value"]}
        for turn in example["conversations"]
    ]
    system_prompt = example.get("system_prompt")
    if system_prompt and (not messages or messages[0]["role"] != "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return {"messages": messages, "source": source}


def convert_messages(example, source):
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in example["messages"]]
    return {"messages": messages, "source": source}


def convert_pair(example, source):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ],
        "source": source,
    }


def output_features():
    from datasets import Features, Value

    return Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "source": Value("string"),
        }
    )


def convert(dataset, function, source, features):
    return dataset.map(
        function,
        fn_kwargs={"source": source},
        remove_columns=dataset.column_names,
        features=features,
        desc=f"Converting {source}",
    )


def build_dataset():
    from datasets import concatenate_datasets, load_dataset

    features = output_features()
    no_robots = load_dataset(
        NO_ROBOTS.repo,
        split="train+test",
        revision=NO_ROBOTS.revision,
    )
    coconot = load_dataset(
        COCONOT.repo,
        "original",
        split="train",
        revision=COCONOT.revision,
    )
    magpie_samples = TOTAL_SAMPLES - OPENHERMES_SAMPLES - len(no_robots) - len(coconot)

    openhermes = load_dataset(
        OPENHERMES.repo,
        split="train",
        revision=OPENHERMES.revision,
    ).shuffle(seed=SEED)
    magpie = load_dataset(
        MAGPIE.repo,
        split="train",
        revision=MAGPIE.revision,
    ).shuffle(seed=SEED)

    datasets = [
        convert(
            openhermes.select(range(OPENHERMES_SAMPLES)),
            convert_sharegpt,
            OPENHERMES.repo,
            features,
        ),
        convert(
            magpie.select(range(magpie_samples)),
            convert_sharegpt,
            MAGPIE.repo,
            features,
        ),
        convert(no_robots, convert_messages, NO_ROBOTS.repo, features),
        convert(coconot, convert_pair, COCONOT.repo, features),
    ]
    mixed = concatenate_datasets(datasets).shuffle(seed=SEED).flatten_indices()
    if len(mixed) != TOTAL_SAMPLES:
        raise RuntimeError(f"expected {TOTAL_SAMPLES:,} rows, built {len(mixed):,}")
    return mixed, datasets


def main():
    mixed, datasets = build_dataset()

    print(f"Publishing {len(mixed):,} samples to {OUTPUT_REPO}")
    for source, dataset in zip(SOURCES, datasets):
        print(f"  {source.repo}@{source.revision}: {len(dataset):,}")
    mixed.push_to_hub(OUTPUT_REPO, split="train")


if __name__ == "__main__":
    main()
