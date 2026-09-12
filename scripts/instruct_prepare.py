"""Compile a configurable English prompt/completion dataset for instruction-tuning pilots."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from speck.chat import ChatFormatError, ChatTokenizer, validate_messages
from speck.io import atomic_json, file_sha256
from speck.tokenizer import Tokenizer

DEFAULT_RECIPE = Path("experiments/Speck-Instruct-Starter/mixture.json")
KINDS = {"messages", "pair", "lmsys", "edit", "science", "table", "constraints"}
ROLES = {
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}
GREETINGS = {"hi", "hello", "hey", "hi there", "hello there"}
FENCES = re.compile(r"```.*?```", re.S)
THINKING = re.compile(r"</?(?:think|analysis|scratchpad|inner_monologue|reasoning)\b", re.I)
UNRESOLVED = re.compile(
    r"\[(?:keyword|relation|frequency|num_placeholders|ender|num_words|num_sentences|num_paragraphs)\]",
    re.I,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def field(row, name, default=None):
    for part in name.split("."):
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except ValueError:
                return default
        if not isinstance(row, dict) or part not in row:
            return default
        row = row[part]
    return row if row is not None else default


def quotas(sources, total, weight_key="weight"):
    """Largest-remainder allocation keeps arbitrary pilot sizes exact."""
    weights = [source.get(weight_key, source["weight"]) for source in sources]
    denominator = sum(weights)
    if not total:
        return {source["id"]: 0 for source in sources}
    if not denominator:
        raise ValueError("positive sample counts require a nonzero allocation weight")
    counts = [total * weight // denominator for weight in weights]
    order = sorted(range(len(sources)), key=lambda i: (-(total * weights[i] % denominator), i))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return dict(zip((source["id"] for source in sources), counts, strict=False))


def load_recipe(path, samples=None, validation_samples=None):
    recipe = json.loads(Path(path).read_text())
    required = {
        "samples",
        "validation_samples",
        "seed",
        "max_tokens",
        "shuffle_buffer",
        "max_source_rows",
        "english_probability",
        "near_duplicate_threshold",
        "tokenizer",
        "sources",
    }
    if set(recipe) != required:
        raise ValueError(f"recipe keys must be {sorted(required)}")
    if type(recipe["samples"]) is not int or recipe["samples"] < 1:
        raise ValueError("samples must be a positive integer")
    if type(recipe["validation_samples"]) is not int or recipe["validation_samples"] < 0:
        raise ValueError("validation_samples must be a nonnegative integer")
    if samples is not None:
        if type(samples) is not int or samples < 1:
            raise ValueError("samples must be a positive integer")
        if validation_samples is None:
            recipe["validation_samples"] = round(
                recipe["validation_samples"] * samples / recipe["samples"]
            )
        recipe["samples"] = samples
    if validation_samples is not None:
        recipe["validation_samples"] = validation_samples
    for key in ("samples", "max_tokens", "shuffle_buffer", "max_source_rows"):
        if type(recipe.get(key)) is not int or recipe[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "validation_samples"):
        if type(recipe.get(key)) is not int or recipe[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("english_probability", "near_duplicate_threshold"):
        value = recipe.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty list")
    seen = set()
    for source in sources:
        allowed = {
            "id",
            "repo",
            "revision",
            "config",
            "split",
            "weight",
            "validation_weight",
            "kind",
            "skill",
            "messages_field",
            "prompt_field",
            "response_field",
            "id_field",
            "family_field",
            "category_field",
            "category_cap",
            "include",
            "minimum",
            "verification",
        }
        if set(source) - allowed:
            raise ValueError(f"unknown source settings: {sorted(set(source) - allowed)}")
        if not re.fullmatch(r"[a-z0-9_-]+", source.get("id", "")) or source["id"] in seen | {
            "train",
            "validation",
            "identity",
            "summary",
            "shortfall",
        }:
            raise ValueError("source IDs must be unique filesystem-safe names")
        seen.add(source["id"])
        if type(source.get("weight")) is not int or source["weight"] <= 0:
            raise ValueError("source weights must be positive integers")
        if "validation_weight" in source and (
            type(source["validation_weight"]) is not int or source["validation_weight"] < 0
        ):
            raise ValueError("validation weights must be nonnegative integers")
        if source.get("kind") not in KINDS:
            raise ValueError(f"unknown adapter: {source.get('kind')}")
        for key in ("repo", "split", "skill"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise ValueError(f"source requires {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
            raise ValueError("source revisions must be full Hub commit hashes")
        if source["kind"] == "pair" and not all(
            source.get(k) for k in ("prompt_field", "response_field")
        ):
            raise ValueError("pair adapter requires prompt_field and response_field")
        for name, values in source.get("include", {}).items():
            if not isinstance(name, str) or not isinstance(values, list) or not values:
                raise ValueError("include filters require field names and nonempty value lists")
        cap = source.get("category_cap", 1.0)
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 1:
            raise ValueError("category_cap must be in (0, 1]")
    quotas(sources, recipe["validation_samples"], "validation_weight")
    if not re.fullmatch(r"[0-9a-f]{40}", recipe["tokenizer"].get("revision", "")):
        raise ValueError("reference tokenizer revision must be a full commit hash")
    return recipe


def adapt(row, source):
    """Small schema adapters; source-specific selection stays in the recipe."""
    for name, allowed in source.get("include", {}).items():
        if field(row, name) not in allowed:
            raise ValueError("source filter")
    for name, minimum in source.get("minimum", {}).items():
        score = field(row, name)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < minimum
        ):
            raise ValueError("source score")
    kind = source["kind"]
    if kind == "lmsys":
        if row.get("grounded") and row.get("agreement") is not True:
            raise ValueError("answer disagreement")
        messages = [
            {"role": "user", "content": row["conversations"][0]["value"]},
            {"role": "assistant", "content": row["deepseek_response"]["value"]},
        ]
    elif kind == "pair":
        messages = [
            {"role": "user", "content": row[source["prompt_field"]]},
            {"role": "assistant", "content": row[source["response_field"]]},
        ]
    elif kind == "edit":
        messages = row["context"] + [{"role": "assistant", "content": row["edited_response"]}]
    else:
        if kind == "science" and not row.get("dataset", "").startswith("science."):
            raise ValueError("non-science replay")
        messages = row[source.get("messages_field", "messages")]
    eligibility = field(row, "metadata.train_turns")
    if eligibility is not None and (
        len(eligibility) != len(messages) or eligibility[-1] is not True
    ):
        raise ValueError("final response is not a training target")
    clean = []
    for message in messages:
        if message.get("tool_calls") or message.get("function_calls") or message.get("functions"):
            raise ValueError("unsupported tool conversation")
        role = ROLES.get(message.get("role", message.get("from")))
        text = message.get("content", message.get("value"))
        if role == "system" and not clean and text == "":
            continue
        if not role or not isinstance(text, str):
            raise ValueError("missing role or content")
        text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
        if kind == "edit" and role == "assistant":
            text = re.sub(r"(?m)(^```(?!text\b|markdown\b)[^\n]*\n)Copy code[ \t]*\n", r"\1", text)
        if "\x00" in text or THINKING.search(text):
            raise ValueError("control or thinking markup")
        clean.append({"role": role, "content": text})
    validate_messages(clean)
    if clean[-1]["role"] != "assistant":
        raise ValueError("missing final assistant")
    validate_messages(clean[:-1], add_generation_prompt=True)
    if kind == "constraints":
        check_simple_constraints(clean[-2]["content"], clean[-1]["content"])
    return clean


def check_simple_constraints(prompt, response):
    """Check unfilled templates and a small literal subset; not a complete IF verifier."""
    if UNRESOLVED.search(prompt):
        raise ValueError("unresolved constraint template")
    lowered = prompt.casefold()
    if "all lowercase" in lowered or "no capital letters" in lowered:
        if any(char.isupper() for char in response):
            raise ValueError("lowercase constraint")
    if "no commas" in lowered and "," in response:
        raise ValueError("comma constraint")
    count = len(response.split())
    for relation, limit in re.findall(
        r"\b(?:contain|use|write|in|with|have|be)\s+(exactly|at least|at most|less than|more than)\s+(\d+)\s+words\b",
        lowered,
    ):
        limit = int(limit)
        valid = {
            "exactly": count == limit,
            "at least": count >= limit,
            "at most": count <= limit,
            "less than": count < limit,
            "more than": count > limit,
        }
        if not valid[relation]:
            raise ValueError("word-count constraint")


def prompt_text(messages):
    """Ignore generic greeting prefixes when identifying a conversation family."""
    users = [message["content"] for message in messages if message["role"] == "user"]
    return next(
        (text for text in users if normalized(text).strip(".!?") not in GREETINGS), users[0]
    )


class Deduplicator:
    """Reject exact prompt/family repeats and high-overlap prompt variants across both splits."""

    def __init__(self, threshold):
        from datasketch import MinHashLSH

        self.threshold = threshold
        self.index = MinHashLSH(threshold=threshold, num_perm=64)
        self.families = set()
        self.shingles = {}
        self.excluded_prompts = set()

    def prepare(self, text):
        from datasketch import MinHash

        words = normalized(text).split()
        # Short instructions need exact matching, not a noisy near-duplicate estimate.
        shingles = (
            {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}
            if len(words) >= 20
            else set()
        )
        signature = None
        if shingles:
            signature = MinHash(num_perm=64)
            signature.update_batch([shingle.encode() for shingle in sorted(shingles)])
        return shingles, signature

    def duplicate(self, families, prepared):
        if self.families.intersection(families):
            return True
        shingles, signature = prepared
        if signature is not None:
            for key in self.index.query(signature):
                other = self.shingles[key]
                if len(shingles & other) / len(shingles | other) >= self.threshold:
                    return True
        return False

    def add(self, families, prepared):
        self.families.update(families)
        shingles, signature = prepared
        if signature is not None:
            key = str(len(self.shingles))
            self.index.insert(key, signature)
            self.shingles[key] = shingles


class EnglishFilter:
    def __init__(self, probability):
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        self.identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        self.probability = probability

    def __call__(self, messages):
        checked = False
        for message in messages:
            if message["role"] == "system":
                continue
            prose = FENCES.sub(" ", message["content"])
            prose = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$|https?://\S+", " ", prose, flags=re.S)
            # Labels, formulas and code-only answers inherit the conversation language.
            if sum(char.isalpha() for char in prose) < 40:
                continue
            language, confidence = self.identifier.classify(prose[:6000])
            if language != "en" or confidence < self.probability:
                return False
            checked = True
        if not checked:
            text = " ".join(
                FENCES.sub(" ", message["content"])
                for message in messages
                if message["role"] != "system"
            )
            language, confidence = self.identifier.classify(text)
            return language == "en" and confidence >= self.probability
        return True


def check_response(text):
    prose = FENCES.sub(" ", text)
    words = normalized(prose).split()
    if len(words) >= 80:
        grams = [tuple(words[i : i + 8]) for i in range(len(words) - 7)]
        if max(Counter(grams).values()) >= 5:
            raise ValueError("repetitive response")
    if text.count("```") % 2:
        raise ValueError("unclosed code block")


def candidate(row, source, index, tokenizer, english):
    messages = adapt(row, source)
    if sum(len(message["content"]) for message in messages) > 200_000:
        raise ValueError("oversized source text")
    check_response(messages[-1]["content"])
    if not english(messages):
        raise ValueError("non-English prose")
    tokens, _ = tokenizer.encode_messages(messages)
    prefix, _ = tokenizer.encode_messages(messages[:-1], add_generation_prompt=True)
    text = prompt_text(messages)
    family = digest(normalized(text))
    source_id = str(field(row, source.get("id_field", "id"), index))
    explicit_family = (
        field(row, source.get("family_field", "")) if source.get("family_field") else None
    )
    family_ids = ["prompt:" + family]
    if explicit_family is not None:
        family_ids.append("source:" + digest([source["repo"], explicit_family]))
    stratum = (
        str(field(row, source.get("category_field", ""), "unknown"))
        if source.get("category_field")
        else "unknown"
    )
    return {
        "id": digest(
            [
                source["repo"],
                source["revision"],
                source.get("config"),
                source["split"],
                source_id,
                messages,
            ]
        ),
        "prompt": messages[:-1],
        "completion": messages[-1:],
        "source": source["repo"],
        "source_id": source_id,
        "source_row_index": index,
        "source_key": source["id"],
        "source_revision": source["revision"],
        "family_ids": family_ids,
        "skill": source["skill"],
        "category": stratum,
        "verification": source.get("verification", "not_independently_verified"),
        "input_tokens": len(prefix),
        "target_tokens": len(tokens) - len(prefix) - len(tokenizer.newline_ids),
        "total_tokens": len(tokens),
        "turns": sum(message["role"] == "assistant" for message in messages),
    }


def source_rows(source, recipe):
    from datasets import load_dataset

    stream = load_dataset(
        source["repo"],
        name=source.get("config"),
        revision=source["revision"],
        split=source["split"],
        streaming=True,
    )
    # Attach the upstream stream position before shuffling; it is not the shuffled position.
    stream = stream.map(lambda row, index: {**row, "_speck_row_index": index}, with_indices=True)
    seed = int(digest([recipe["seed"], source["id"]])[:8], 16)
    return stream.shuffle(seed=seed, buffer_size=recipe["shuffle_buffer"])


def select_source(source, recipe, targets, tokenizer, english, dedup, rows):
    counts, rejected, categories = Counter(), Counter(), Counter()
    selected = []
    fraction = recipe["validation_samples"] / (recipe["samples"] + recipe["validation_samples"])
    scanned = 0
    for scanned, row in enumerate(rows, 1):
        if scanned > recipe["max_source_rows"]:
            scanned -= 1
            break
        # Sorted upstream subsets can contain thousands of rows from an already-full category.
        # Skip those before language identification and tokenization, without changing selection.
        category = field(row, source.get("category_field", ""))
        if category is not None and all(
            counts[split] >= amount
            or categories[split, str(category)]
            >= math.ceil(amount * source.get("category_cap", 1.0))
            for split, amount in targets.items()
        ):
            rejected["category cap"] += 1
            continue
        try:
            item = candidate(
                row, source, row.get("_speck_row_index", scanned - 1), tokenizer, english
            )
            if any(
                message["role"] == "user"
                and normalized(message["content"]) in dedup.excluded_prompts
                for message in item["prompt"]
            ):
                raise ValueError("excluded user turn")
            if item["total_tokens"] > recipe["max_tokens"]:
                raise ValueError("over context limit")
            if item["target_tokens"] <= 0:
                raise ValueError("empty target")
            split_hash = digest([recipe["seed"], item["family_ids"][0]])
            split = "validation" if int(split_hash[:16], 16) / 2**64 < fraction else "train"
            if counts[split] >= targets[split]:
                raise ValueError("split quota filled")
            cap = math.ceil(targets[split] * source.get("category_cap", 1.0))
            if item["category"] != "unknown" and categories[split, item["category"]] >= cap:
                raise ValueError("category cap")
            prepared = dedup.prepare(prompt_text(item["prompt"]))
            if dedup.duplicate(item["family_ids"], prepared):
                raise ValueError("duplicate prompt or family")
        except (ValueError, KeyError, IndexError, TypeError, ChatFormatError) as error:
            rejected[str(error)] += 1
            continue
        item["split"] = split
        selected.append(item)
        counts[split] += 1
        categories[split, item["category"]] += 1
        dedup.add(item["family_ids"], prepared)
        if all(counts[split] == amount for split, amount in targets.items()):
            break
    stats = {
        "scanned": scanned,
        "accepted": dict(counts),
        "requested": targets,
        "rejected": dict(rejected),
        "categories": {f"{s}/{c}": n for (s, c), n in categories.items()},
    }
    return selected, stats


def add_exclusions(paths, dedup):
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                messages = row.get("messages", row.get("prompt"))
                text = (
                    prompt_text(messages)
                    if isinstance(messages, list)
                    else row.get("text", messages)
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"exclusion row needs messages, prompt, or text: {path}")
                families = ["prompt:" + digest(normalized(text)), *row.get("family_ids", [])]
                dedup.excluded_prompts.add(normalized(text))
                dedup.add(families, dedup.prepare(text))


def build(
    recipe,
    output,
    *,
    resume=False,
    exclusions=(),
    rows_factory=source_rows,
    tokenizer=None,
    english=None,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if tokenizer is None:
        tokenizer = ChatTokenizer(Tokenizer(hf_hub_download(**recipe["tokenizer"])))
    if english is None:
        english = EnglishFilter(recipe["english_probability"])
    identity = {
        "recipe": recipe,
        "tokenizer": tokenizer.metadata(),
        "exclusions": [
            {"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in exclusions
        ],
        "compiler_sha256": file_sha256(__file__),
        "packages": {
            name: version(name)
            for name in ("datasets", "pyarrow", "py3langid", "datasketch", "sentencepiece")
        },
    }
    building = output.with_name(output.name + ".building")
    if building.exists():
        if not resume or json.loads((building / "identity.json").read_text()) != identity:
            raise ValueError(
                "incomplete build exists; --resume requires the identical recipe, compiler and tokenizer"
            )
    else:
        building.mkdir(parents=True)
        atomic_json(building / "identity.json", identity)
    dedup = Deduplicator(recipe["near_duplicate_threshold"])
    add_exclusions(exclusions, dedup)
    train = quotas(recipe["sources"], recipe["samples"])
    validation = quotas(recipe["sources"], recipe["validation_samples"], "validation_weight")
    tables, reports = [], {}
    for source in recipe["sources"]:
        key = source["id"]
        targets = {"train": train[key], "validation": validation[key]}
        if not sum(targets.values()):
            continue
        data_path, report_path = building / f"{key}.parquet", building / f"{key}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if file_sha256(data_path) != report["sha256"]:
                raise ValueError(f"corrupt completed source: {key}")
            table = pq.read_table(data_path)
            for item in table.to_pylist():
                dedup.add(item["family_ids"], dedup.prepare(prompt_text(item["prompt"])))
            print(f"Reusing {key}: {table.num_rows:,} rows", flush=True)
        else:
            print(f"Selecting {key}: {targets}", flush=True)
            items, report = select_source(
                source, recipe, targets, tokenizer, english, dedup, rows_factory(source, recipe)
            )
            if any(report["accepted"].get(split, 0) != amount for split, amount in targets.items()):
                atomic_json(building / "shortfall.json", {"source": key, **report})
                raise ValueError(
                    f"{key} quota shortfall: {report['accepted']} versus {targets}; see {building / 'shortfall.json'}"
                )
            table = pa.Table.from_pylist(items)
            pq.write_table(table, data_path, compression="zstd")
            report["sha256"] = file_sha256(data_path)
            atomic_json(report_path, report)
            print(f"Selected {key}: {len(items):,} / {report['scanned']:,} scanned", flush=True)
        tables.append(table)
        reports[key] = report
    combined = pa.concat_tables(tables)
    indices = list(range(combined.num_rows))
    random.Random(recipe["seed"]).shuffle(indices)
    combined = combined.take(pa.array(indices))
    import pyarrow.compute as pc

    outputs = {}
    for split in ("train", "validation"):
        table = combined.filter(pc.equal(combined["split"], split)).drop(["split"])
        path = building / f"{split}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=512)
        outputs[split] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "samples": table.num_rows,
            "processed_tokens": sum(table["total_tokens"].to_pylist()),
            "target_tokens": sum(table["target_tokens"].to_pylist()),
            "multi_turn_samples": sum(n > 1 for n in table["turns"].to_pylist()),
            "over_4k_samples": sum(n > 4096 for n in table["total_tokens"].to_pylist()),
            "skills": dict(Counter(table["skill"].to_pylist())),
            "length_buckets": dict(
                Counter(
                    str(
                        next(
                            (limit for limit in (512, 1024, 2048, 4096, 8192) if n <= limit),
                            "over_8k",
                        )
                    )
                    for n in table["total_tokens"].to_pylist()
                )
            ),
        }
    atomic_json(
        building / "summary.json",
        {
            "format": "speck_instruct_starter",
            "version": 1,
            "identity": identity,
            "supervision": "completion_only_final_assistant",
            "sources": reports,
            "outputs": outputs,
            "quality_scope": "Source metadata and heuristic format/language/length/repetition filters; semantic correctness is not independently certified.",
            "deduplication": "Exact first substantive prompt and explicit source families; MinHash candidates verified by 5-word-shingle Jaccard. Not exhaustive semantic deduplication.",
        },
    )
    # Completed per-source files support resume only while a build is unfinished.
    for key in reports:
        (building / f"{key}.parquet").unlink()
        (building / f"{key}.json").unlink()
    (building / "shortfall.json").unlink(missing_ok=True)
    os.replace(building, output)
    return outputs


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, help="override total training conversations")
    parser.add_argument("--validation-samples", type=int, help="additional held-out conversations")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="JSONL prompts/families to exclude",
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse hash-checked completed sources"
    )
    parser.add_argument("--dry-run", action="store_true", help="print quotas without downloading")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    return args


def main(argv=None):
    args = arguments(argv)
    recipe = load_recipe(args.recipe, args.samples, args.validation_samples)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "train": quotas(recipe["sources"], recipe["samples"]),
                    "validation": quotas(
                        recipe["sources"], recipe["validation_samples"], "validation_weight"
                    ),
                },
                indent=2,
            )
        )
        return
    outputs = build(recipe, args.output_dir, resume=args.resume, exclusions=args.exclude)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
