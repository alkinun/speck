#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub>=0.34.0",
#     "sentencepiece>=0.2.0",
# ]
# ///
"""Build and publish the 500K-example SpeckChat2 instruction dataset."""

import argparse
import gc
import hashlib
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

OUTPUT_REPO = "specklabs/SpeckChat2"
SEED = 42
TOTAL_SAMPLES = 500_000
MAX_CONTEXT_TOKENS = 2_048
MAX_ASSISTANT_TOKENS = 1_536

TOKENIZER_REPO = "mistralai/Mistral-7B-v0.1"
TOKENIZER_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}
RESERVED_TOKENS = ("<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>")
WHITESPACE = re.compile(r"\s+")
GENERIC_HERMES_SYSTEM_PROMPTS = {
    "you are a helpful assistant.",
    "you are an unbiased, uncensored, helpful assistant.",
}
EVERYDAY_GREETINGS = {"hello", "hey!", "hi", "hi.", "hi there"}
EVERYDAY_GREETING_RESPONSES = {
    "hello! how can i help you today?",
    "hello. how can i help you today?",
}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    repo: str
    revision: str
    split: str
    quota: int
    declared_license: str
    adapter: object
    quality_bands: tuple = ((None, None),)
    category_caps: tuple = ()
    category_targets: tuple = ()


class SpeckTokenCounter:
    """Measure rows exactly as the Speck chat tokenizer serializes them."""

    def __init__(self, processor):
        self.processor = processor
        self.newline_tokens = processor.encode("\n", out_type=int)

    @classmethod
    def from_hub(cls, cache_dir=None):
        import sentencepiece
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            TOKENIZER_REPO,
            "tokenizer.model",
            revision=TOKENIZER_REVISION,
            cache_dir=cache_dir,
        )
        return cls(sentencepiece.SentencePieceProcessor(model_file=path))

    def measure(self, messages):
        context_tokens = 1  # BOS
        assistant_tokens = 0
        for message in messages:
            content = self.processor.encode("\n" + message["content"], out_type=int)
            context_tokens += 1 + len(content) + 1 + len(self.newline_tokens)
            if message["role"] == "assistant":
                assistant_tokens += len(content) - len(self.newline_tokens) + 1
        return context_tokens, assistant_tokens


def _clean_content(value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = value.strip()
    return value or None


def canonicalize_messages(messages):
    if not isinstance(messages, list):
        return None
    canonical = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = ROLE_MAP.get(message.get("role", message.get("from")))
        content = _clean_content(message.get("content", message.get("value")))
        if role is None or content is None:
            return None
        canonical.append({"role": role, "content": content})
    return canonical


def validate_messages(messages):
    if not messages:
        return "empty conversation"
    system_offset = int(messages[0]["role"] == "system")
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            if index != 0:
                return "system message is not first"
        else:
            expected = "user" if (index - system_offset) % 2 == 0 else "assistant"
            if role != expected:
                return "roles do not alternate"
        if "\x00" in message["content"]:
            return "content contains a null byte"
        if any(token in message["content"] for token in RESERVED_TOKENS):
            return "content contains a reserved token"
    if messages[-1]["role"] != "assistant":
        return "conversation does not end with assistant"
    return None


def _hash_normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return WHITESPACE.sub(" ", value).strip()


def prompt_digest(messages):
    prompt = "\n".join(
        f"{message['role']}\x1f{_hash_normalize(message['content'])}"
        for message in messages
        if message["role"] != "assistant"
    )
    return hashlib.blake2b(prompt.encode(), digest_size=16).digest()


def _stable_id(spec, source_id, source_row_index):
    identity = (
        f"{spec.repo}\x1f{spec.revision}\x1f{spec.split}\x1f{source_id}\x1f{source_row_index}"
    )
    return hashlib.blake2b(identity.encode(), digest_size=16).hexdigest()


def _candidate(
    messages,
    source_id,
    category="unknown",
    difficulty="unknown",
    language="en",
    generator="unknown",
    quality_score=None,
):
    return {
        "messages": messages,
        "source_id": str(source_id),
        "category": str(category or "unknown"),
        "difficulty": str(difficulty or "unknown"),
        "language": str(language or "unknown"),
        "generator": str(generator or "unknown"),
        "quality_score": quality_score,
    }


def adapt_lmsys(example, source_row_index):
    if example.get("flaw") != "normal":
        return None
    if example.get("grounded") and example.get("agreement") is not True:
        return None
    conversations = example.get("conversations") or []
    response = example.get("deepseek_response") or {}
    if not conversations or not response.get("value"):
        return None
    return _candidate(
        [
            {"role": "user", "content": conversations[0].get("value")},
            {"role": "assistant", "content": response["value"]},
        ],
        example.get("id", source_row_index),
        category=example.get("category"),
        generator="DeepSeek-V3",
        quality_score=response.get("reward"),
    )


def adapt_magpie_mt(example, source_row_index):
    if str(example.get("language", "")).upper() != "EN":
        return None
    if example.get("llama_guard_2") != "safe":
        return None
    if example.get("input_quality") not in {"good", "excellent"}:
        return None
    conversations = example.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 4:
        return None
    return _candidate(
        conversations,
        example.get("uuid", source_row_index),
        category=example.get("task_category"),
        difficulty=example.get("difficulty"),
        language="en",
        generator=example.get("model"),
        quality_score=example.get("instruct_reward"),
    )


def adapt_magpie_reasoning(example, source_row_index):
    if str(example.get("language", "")).upper() != "EN":
        return None
    if example.get("input_quality") not in {"good", "excellent"}:
        return None
    if example.get("difficulty") not in {"easy", "medium"}:
        return None
    generation = example.get("gen_response_configs") or {}
    return _candidate(
        [
            {"role": "user", "content": example.get("instruction")},
            {"role": "assistant", "content": example.get("response")},
        ],
        example.get("uuid", source_row_index),
        category=example.get("task_category"),
        difficulty=example.get("difficulty"),
        language="en",
        generator=generation.get("output_generator"),
    )


def adapt_hermes(example, source_row_index):
    messages = canonicalize_messages(example.get("messages"))
    if messages and messages[0]["role"] == "system":
        system_prompt = _hash_normalize(messages[0]["content"])
        if system_prompt in GENERIC_HERMES_SYSTEM_PROMPTS:
            messages = messages[1:]
    return _candidate(messages, source_row_index, generator="mixed")


def adapt_ultrachat(example, source_row_index):
    return _candidate(example.get("messages"), source_row_index, generator="ChatGPT")


def adapt_everyday(example, source_row_index):
    messages = canonicalize_messages(example.get("messages"))
    if messages and len(messages) >= 4:
        first = _hash_normalize(messages[0]["content"])
        second = _hash_normalize(messages[1]["content"])
        if first in EVERYDAY_GREETINGS and second in EVERYDAY_GREETING_RESPONSES:
            messages = messages[2:]
    return _candidate(
        messages,
        example.get("full_topic", source_row_index),
        category=example.get("topic"),
        generator="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )


def adapt_no_robots(example, source_row_index):
    return _candidate(
        example.get("messages"),
        example.get("prompt_id", source_row_index),
        category=example.get("category"),
        generator="human",
    )


SOURCES = (
    SourceSpec(
        key="lmsys",
        repo="OpenLeecher/lmsys_chat_1m_clean",
        revision="e9f2f6838a2dbba87c216bb6bc406e8d7ce0f389",
        split="train",
        quota=200_000,
        declared_license="unspecified",
        adapter=adapt_lmsys,
        quality_bands=((0.0, None), (-2.0, 0.0), (None, -2.0)),
    ),
    SourceSpec(
        key="magpie_mt",
        repo="Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1",
        revision="266269affc19d473119d77a247050ab8a75db376",
        split="train",
        quota=130_000,
        declared_license="unspecified; subject to Llama 3.1 terms",
        adapter=adapt_magpie_mt,
        quality_bands=((0.0, None), (-2.0, 0.0)),
        category_caps=(("Math", 26_000), ("Coding & Debugging", 20_000)),
    ),
    SourceSpec(
        key="hermes",
        repo="enPurified/Hermes-3-Dataset-enPurified-openai-messages",
        revision="6c5419b947849bf4e6b66db0bb0a14f47926af78",
        split="train",
        quota=85_000,
        declared_license="other",
        adapter=adapt_hermes,
    ),
    SourceSpec(
        key="ultrachat",
        repo="enPurified/ultrachat_200k_sft-enPurified-openai-messages",
        revision="37ebbe63f5c244d8899f438f5fb75daa7f853c73",
        split="train",
        quota=65_000,
        declared_license="unknown",
        adapter=adapt_ultrachat,
    ),
    SourceSpec(
        key="magpie_reasoning",
        repo="Magpie-Align/Magpie-Reasoning-V1-150K",
        revision="a4bedadca568ba8fa50cae618ae62ca34dd1d196",
        split="train",
        quota=10_000,
        declared_license="llama3; subject to upstream terms",
        adapter=adapt_magpie_reasoning,
        category_targets=(("Math", 4_000), ("Coding & Debugging", 4_000), ("Reasoning", 2_000)),
    ),
    SourceSpec(
        key="no_robots",
        repo="HuggingFaceH4/no_robots",
        revision="e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
        split="train",
        quota=8_000,
        declared_license="cc-by-nc-4.0",
        adapter=adapt_no_robots,
        category_caps=(("Generation", 3_000),),
    ),
    SourceSpec(
        key="everyday",
        repo="HuggingFaceTB/everyday-conversations-llama3.1-2k",
        revision="14f543216b9ba42b6b951dc5bd199460d193b162",
        split="train_sft",
        quota=2_000,
        declared_license="apache-2.0",
        adapter=adapt_everyday,
    ),
)
SOURCE_BY_KEY = {source.key: source for source in SOURCES}
PROCESSING_ORDER = (
    "no_robots",
    "everyday",
    "lmsys",
    "hermes",
    "magpie_reasoning",
    "magpie_mt",
    "ultrachat",
)


def source_quality_score(spec, example):
    if spec.key == "lmsys":
        return (example.get("deepseek_response") or {}).get("reward")
    if spec.key == "magpie_mt":
        return example.get("instruct_reward")
    return None


def _in_quality_band(score, band, band_index):
    lower, upper = band
    if score is None:
        return band_index == 0
    return (lower is None or score >= lower) and (upper is None or score < upper)


def _source_seed(spec):
    value = hashlib.blake2b(f"{SEED}:{spec.repo}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "little")


def prepare_candidate(spec, candidate, source_row_index, token_counter):
    if candidate is None:
        return None, None, "source filter"
    messages = canonicalize_messages(candidate.get("messages"))
    error = validate_messages(messages)
    if error:
        return None, None, error
    context_tokens, assistant_tokens = token_counter.measure(messages)
    if context_tokens > MAX_CONTEXT_TOKENS:
        return None, None, "over context length"
    if assistant_tokens > MAX_ASSISTANT_TOKENS:
        return None, None, "too many assistant tokens"
    if assistant_tokens < 2:
        return None, None, "no assistant target"

    source_id = candidate["source_id"]
    row = {
        "id": _stable_id(spec, source_id, source_row_index),
        "messages": messages,
        "source": spec.repo,
        "source_revision": spec.revision,
        "source_split": spec.split,
        "source_id": source_id,
        "source_row_index": source_row_index,
        "declared_license": spec.declared_license,
        "generator": candidate["generator"],
        "category": candidate["category"],
        "difficulty": candidate["difficulty"],
        "language": candidate["language"],
        "quality_score": candidate["quality_score"],
        "turns": sum(message["role"] == "assistant" for message in messages),
        "context_tokens": context_tokens,
        "assistant_tokens": assistant_tokens,
    }
    return row, prompt_digest(messages), None


def select_source(dataset, spec, token_counter, seen_prompts):
    indices = list(range(len(dataset)))
    random.Random(_source_seed(spec)).shuffle(indices)
    selected = []
    rejections = Counter()
    categories = Counter()
    category_caps = dict(spec.category_caps)
    category_targets = dict(spec.category_targets)

    for band_index, band in enumerate(spec.quality_bands):
        for source_row_index in indices:
            if len(selected) == spec.quota:
                break
            example = dataset[source_row_index]
            score = source_quality_score(spec, example)
            if not _in_quality_band(score, band, band_index):
                continue
            candidate = spec.adapter(example, source_row_index)
            if candidate is None:
                rejections["source filter"] += 1
                continue
            category = candidate["category"]
            if category_targets:
                if category not in category_targets:
                    rejections["category not selected"] += 1
                    continue
                if categories[category] >= category_targets[category]:
                    rejections["category quota filled"] += 1
                    continue
            if category in category_caps and categories[category] >= category_caps[category]:
                rejections["category cap"] += 1
                continue

            row, digest, error = prepare_candidate(spec, candidate, source_row_index, token_counter)
            if error:
                rejections[error] += 1
                continue
            if digest in seen_prompts:
                rejections["duplicate prompt"] += 1
                continue
            seen_prompts.add(digest)
            selected.append(row)
            categories[category] += 1
        if len(selected) == spec.quota:
            break

    if len(selected) != spec.quota:
        raise RuntimeError(
            f"{spec.repo} produced {len(selected):,} of {spec.quota:,} required rows; "
            f"rejections={dict(rejections)}"
        )
    for category, target in category_targets.items():
        if categories[category] != target:
            raise RuntimeError(
                f"{spec.repo} category {category!r} produced {categories[category]:,} "
                f"of {target:,} required rows"
            )
    return selected, rejections, categories


def output_features():
    from datasets import Features, Value

    return Features(
        {
            "id": Value("string"),
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "source": Value("string"),
            "source_revision": Value("string"),
            "source_split": Value("string"),
            "source_id": Value("string"),
            "source_row_index": Value("int64"),
            "declared_license": Value("string"),
            "generator": Value("string"),
            "category": Value("string"),
            "difficulty": Value("string"),
            "language": Value("string"),
            "quality_score": Value("float32"),
            "turns": Value("int16"),
            "context_tokens": Value("int32"),
            "assistant_tokens": Value("int32"),
        }
    )


def build_dataset(cache_dir=None):
    from datasets import Dataset, concatenate_datasets, load_dataset

    if sum(source.quota for source in SOURCES) != TOTAL_SAMPLES:
        raise RuntimeError("source quotas do not sum to TOTAL_SAMPLES")

    token_counter = SpeckTokenCounter.from_hub(cache_dir=cache_dir)
    features = output_features()
    seen_prompts = set()
    selected_datasets = {}

    for key in PROCESSING_ORDER:
        spec = SOURCE_BY_KEY[key]
        print(f"Loading {spec.repo}@{spec.revision} ({spec.split})")
        dataset = load_dataset(
            spec.repo,
            split=spec.split,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
        rows, rejections, categories = select_source(dataset, spec, token_counter, seen_prompts)
        selected_datasets[key] = Dataset.from_list(rows, features=features)
        print(f"Selected {len(rows):,} rows from {spec.repo}")
        print(f"  categories: {dict(categories.most_common())}")
        print(f"  rejections: {dict(rejections.most_common())}")
        del dataset, rows
        gc.collect()

    mixed = concatenate_datasets([selected_datasets[source.key] for source in SOURCES]).shuffle(
        seed=SEED
    )
    mixed = mixed.flatten_indices()
    if len(mixed) != TOTAL_SAMPLES:
        raise RuntimeError(f"expected {TOTAL_SAMPLES:,} rows, built {len(mixed):,}")
    actual = Counter(mixed["source"])
    expected = {source.repo: source.quota for source in SOURCES}
    if dict(actual) != expected:
        raise RuntimeError(f"source count mismatch: expected {expected}, built {dict(actual)}")
    return mixed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-repo", default=OUTPUT_REPO, help="Hugging Face dataset repo")
    parser.add_argument("--output-dir", type=Path, help="optional local save_to_disk directory")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument("--private", action="store_true", help="make a newly created repo private")
    parser.add_argument("--no-push", action="store_true", help="build locally without uploading")
    args = parser.parse_args()
    if args.no_push and args.output_dir is None:
        parser.error("--no-push requires --output-dir")
    return args


def main():
    args = parse_args()
    mixed = build_dataset(cache_dir=args.cache_dir)
    print(f"Built {len(mixed):,} SpeckChat2 training samples")

    if args.output_dir is not None:
        if args.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        mixed.save_to_disk(args.output_dir)
        print(f"Saved dataset to {args.output_dir}")
    if not args.no_push:
        print(f"Publishing train split to {args.output_repo}")
        mixed.push_to_hub(
            args.output_repo,
            split="train",
            private=args.private,
            max_shard_size="500MB",
        )


if __name__ == "__main__":
    main()
