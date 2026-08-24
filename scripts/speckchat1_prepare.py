#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=4.0.0",
# ]
# ///
"""Build and publish the 300K-example SpeckChat1 instruction dataset."""

from datasets import Features, Value, concatenate_datasets, load_dataset

OUTPUT_REPO = "specklabs/SpeckChat1"
SEED = 42
TOTAL_SAMPLES = 300_000
OPENHERMES_SAMPLES = 180_000

OPENHERMES = "teknium/OpenHermes-2.5"
MAGPIE = "Magpie-Align/Magpie-Air-MT-300K-v0.1"
NO_ROBOTS = "HuggingFaceH4/no_robots"
COCONOT = "allenai/coconot"

FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "source": Value("string"),
    }
)
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
    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in example["messages"]
    ]
    return {"messages": messages, "source": source}


def convert_pair(example, source):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ],
        "source": source,
    }


def convert(dataset, function, source):
    return dataset.map(
        function,
        fn_kwargs={"source": source},
        remove_columns=dataset.column_names,
        features=FEATURES,
        desc=f"Converting {source}",
    )


def main():
    no_robots = load_dataset(NO_ROBOTS, split="train+test")
    coconot = load_dataset(COCONOT, "original", split="train")
    magpie_samples = TOTAL_SAMPLES - OPENHERMES_SAMPLES - len(no_robots) - len(coconot)

    openhermes = load_dataset(OPENHERMES, split="train").shuffle(seed=SEED)
    magpie = load_dataset(MAGPIE, split="train").shuffle(seed=SEED)

    datasets = [
        convert(openhermes.select(range(OPENHERMES_SAMPLES)), convert_sharegpt, OPENHERMES),
        convert(magpie.select(range(magpie_samples)), convert_sharegpt, MAGPIE),
        convert(no_robots, convert_messages, NO_ROBOTS),
        convert(coconot, convert_pair, COCONOT),
    ]
    mixed = concatenate_datasets(datasets).shuffle(seed=SEED).flatten_indices()

    print(f"Publishing {len(mixed):,} samples to {OUTPUT_REPO}")
    for source, dataset in zip((OPENHERMES, MAGPIE, NO_ROBOTS, COCONOT), datasets):
        print(f"  {source}: {len(dataset):,}")
    mixed.push_to_hub(OUTPUT_REPO, split="train")


if __name__ == "__main__":
    main()
