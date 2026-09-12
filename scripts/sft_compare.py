"""Compare a local SFT checkpoint with a pinned public Instruct release on fixed pilot tasks."""

import argparse
import gc
import json
import random
from dataclasses import fields
from itertools import permutations
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from scripts.infer import load_checkpoint_model
from scripts.instruct_eval import QUESTIONS, score
from speck.architecture import ArchitectureConfig
from speck.chat import get_chat_tokenizer
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.generation import generate_tokens
from speck.io import atomic_json, file_sha256
from speck.model import SpeckForCausalLM
from speck.sft import (
    load_sft_manifest,
    resolve_sft_data_dir,
    sft_loader,
    sft_plan,
    validate_sft,
    verify_sft_dataset,
)


def questions(seed=20260911, per_task=40):
    """Construct pilot cases before model outputs; these are not an official benchmark."""
    rng = random.Random(seed)
    cases = [{"id": f"legacy-{i}", **item} for i, item in enumerate(QUESTIONS)]
    words = [
        "amber",
        "birch",
        "cedar",
        "cobalt",
        "coral",
        "elm",
        "fern",
        "jade",
        "maple",
        "oak",
        "pearl",
        "pine",
    ]
    pairs = rng.sample([(a, b) for a in range(10, 81) for b in range(1, 20)], per_task)
    orderings = rng.sample(list(permutations(words, 4)), per_task)
    phrases = rng.sample(list(permutations(words, 3)), per_task)
    for i, (a, b) in enumerate(pairs):
        selected = orderings[i]
        records = {name: str(rng.randint(100, 999)) for name in rng.sample(words, 5)}
        key = rng.choice(list(records))
        phrase = " ".join(phrases[i])
        cases.extend(
            [
                {
                    "id": f"arithmetic-{i}",
                    "category": "arithmetic",
                    "prompt": f"Calculate {a} - {b}. Answer with only the number.",
                    "accepted": [str(a - b)],
                    "scoring": "final_number",
                },
                {
                    "id": f"sorting-{i}",
                    "category": "sorting",
                    "prompt": f"Sort these words alphabetically: {', '.join(selected)}. Return only the words separated by commas.",
                    "accepted": [", ".join(sorted(selected))],
                },
                {
                    "id": f"grounding-{i}",
                    "category": "grounding",
                    "prompt": "Use only this inventory:\n"
                    + "\n".join(f"{name}: {value}" for name, value in records.items())
                    + f"\nWhat is the inventory number for {key}? Answer with only the number.",
                    "accepted": [records[key]],
                    "scoring": "final_number",
                },
                {
                    "id": f"format-{i}",
                    "category": "format",
                    "prompt": f"Reply with exactly these three words and nothing else: {phrase}",
                    "accepted": [phrase],
                    "scoring": "literal",
                },
                {
                    "id": f"code-{i}",
                    "category": "code_trace",
                    "prompt": f"What does this Python code print?\nvalues = [{a}, {b}, 3]\nprint(values[1] + len(values))\nAnswer with only the number.",
                    "accepted": [str(b + 3)],
                    "scoring": "final_number",
                },
            ]
        )
    qualitative = [
        [{"role": "user", "content": "Hello! What can you help me with?"}],
        [{"role": "user", "content": "Explain why the sky looks blue in two short sentences."}],
        [
            {
                "role": "user",
                "content": "Rewrite this politely: Send me the report now. You are late.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Summarize in one sentence: The meeting moved from Tuesday to Thursday because the venue was unavailable. The start time remains 10 a.m.",
            }
        ],
        [{"role": "user", "content": "Suggest three inexpensive vegetarian lunches."}],
        [
            {
                "role": "user",
                "content": "Write a Python function that returns the largest number in a nonempty list.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Why does this fail, and how do I fix it?\nname = 'Ada'\nprint(name + 3)",
            }
        ],
        [{"role": "user", "content": "Return JSON with name Ada and age 28. Output only JSON."}],
        [
            {
                "role": "user",
                "content": "Document: The library opens at 9 a.m. on weekdays.\nWhat time does it close? Use only the document.",
            }
        ],
        [{"role": "user", "content": "Which is larger, 0.8 or 0.75? Briefly explain."}],
        [
            {"role": "user", "content": "My budget is 30 dollars. Suggest a birthday gift."},
            {"role": "assistant", "content": "A book or a small plant could fit that budget."},
            {
                "role": "user",
                "content": "They dislike plants and love cooking. Revise your suggestion.",
            },
        ],
        [
            {"role": "system", "content": "Respond in one short sentence."},
            {"role": "user", "content": "Explain what a computer program is."},
        ],
    ]
    return cases, qualitative


def load_baseline(repo, revision, tokenizer, device):
    config_path = hf_hub_download(repo, "config.json", revision=revision)
    weights_path = hf_hub_download(repo, "model.safetensors", revision=revision)
    tokenizer_path = hf_hub_download(repo, "tokenizer.model", revision=revision)
    chat_path = hf_hub_download(repo, "tokenizer_config.json", revision=revision)
    if file_sha256(tokenizer_path) != tokenizer.base.fingerprint():
        raise ValueError("baseline tokenizer differs from candidate tokenizer")
    if (
        json.loads(Path(chat_path).read_text()).get("chat_template")
        != tokenizer.metadata()["chat_template"]
    ):
        raise ValueError("baseline chat serialization differs from candidate")
    allowed = {field.name for field in fields(ArchitectureConfig)}
    config = ArchitectureConfig.from_dict(
        {
            key: value
            for key, value in json.loads(Path(config_path).read_text()).items()
            if key in allowed
        }
    )
    model = SpeckForCausalLM(config)
    state = load_file(weights_path)
    if all(key.startswith("native.") for key in state):
        state = {key.removeprefix("native."): value for key, value in state.items()}
    state.setdefault("lm_head.weight", state["embed_tokens.weight"])
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval(), {
        "repo": repo,
        "revision": revision,
        "weights_sha256": file_sha256(weights_path),
        "config_sha256": file_sha256(config_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }


def generate(model, tokenizer, messages, max_tokens, device):
    tokens, _ = tokenizer.encode_messages(messages, add_generation_prompt=True)
    generated = generate_tokens(
        model, tokens, max_tokens=max_tokens, eos_token_id=tokenizer.eos_id, device=device
    )
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


def evaluate(model, identity, tokenizer, cases, qualitative, data_dir, max_tokens, device):
    answers = []
    for index, case in enumerate(cases):
        answer, count = generate(
            model, tokenizer, [{"role": "user", "content": case["prompt"]}], max_tokens, device
        )
        if case.get("scoring") == "literal":
            correct = exact = answer in case["accepted"]
        else:
            correct, exact = score(answer, case["accepted"], case.get("scoring", "contains"))
        answers.append(
            {
                **case,
                "answer": answer,
                "generated_tokens": count,
                "correct": correct,
                "exact_format": exact,
            }
        )
        if (index + 1) % 40 == 0:
            print(f"Evaluated {index + 1}/{len(cases)} scored cases", flush=True)
    conversations = []
    for messages in qualitative:
        answer, count = generate(model, tokenizer, messages, max_tokens, device)
        conversations.append({"messages": messages, "answer": answer, "generated_tokens": count})
    manifest = load_sft_manifest(data_dir)
    plan = sft_plan(manifest, "val", 8192)
    loss, targets = validate_sft(
        model,
        sft_loader(tokenizer, 8192, split="val", device=device, data_dir=data_dir),
        plan["cycle_microbatches"],
    )
    categories = {}
    for item in answers:
        entry = categories.setdefault(
            item["category"], {"correct": 0, "exact_format": 0, "count": 0}
        )
        entry["correct"] += item["correct"]
        entry["exact_format"] += item["exact_format"]
        entry["count"] += 1
    return {
        "identity": identity,
        "correct": sum(item["correct"] for item in answers),
        "exact_format": sum(item["exact_format"] for item in answers),
        "count": len(answers),
        "categories": categories,
        "starter_validation_loss": loss,
        "validation_target_tokens": targets,
        "answers": answers,
        "conversations": conversations,
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="prepared SFT data for common held-out loss when comparing different training corpora",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write fixed cases before training; do not load models",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    configs = load_experiment(args.experiment, "sft", "tokenizer", "comparison")
    comparison = configs["comparison"]
    cases, qualitative = questions(comparison["seed"], comparison["cases_per_task"])
    payload = {"cases": cases, "qualitative": qualitative, "settings": comparison}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_path = output / "cases.json"
    if case_path.exists() and json.loads(case_path.read_text()) != json.loads(json.dumps(payload)):
        raise ValueError("output directory is bound to different evaluation cases")
    if not case_path.exists():
        atomic_json(case_path, payload)
    if args.prepare_only:
        print(f"Prepared {len(cases)} scored and {len(qualitative)} qualitative cases: {case_path}")
        return
    report_path = output / "comparison.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    tokenizer = get_chat_tokenizer(**configs["tokenizer"])
    data_dir = resolve_sft_data_dir(configs["sft"]["dataset"], configs["sft"].get("data_dir"))
    verify_sft_dataset(data_dir, load_sft_manifest(data_dir))
    training_manifest_sha256 = file_sha256(data_dir / "manifest.json")
    if args.validation_data_dir is not None:
        data_dir = args.validation_data_dir.expanduser().resolve()
        manifest = load_sft_manifest(data_dir)
        if manifest["tokenizer"] != tokenizer.metadata():
            raise ValueError("validation dataset tokenizer mismatch")
        verify_sft_dataset(data_dir, manifest)
    checkpoint_dir = args.checkpoint_dir or Path(base_dir()) / "checkpoints" / configs["sft"]["run"]
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError("candidate checkpoint is not complete")
    device = torch.device(args.device)
    torch.set_num_threads(8)
    torch.manual_seed(comparison["seed"])
    results = {}
    for name in ("released", "starter"):
        if name == "released":
            model, identity = load_baseline(
                comparison["baseline_repo"], comparison["baseline_revision"], tokenizer, device
            )
        else:
            model, metadata = load_checkpoint_model(checkpoint_dir, step, "cpu")
            if metadata["resolved"]["tokenizer"] != tokenizer.metadata():
                raise ValueError("candidate checkpoint tokenizer mismatch")
            model = model.to(device=device, dtype=torch.bfloat16).eval()
            identity = checkpoint_identity(checkpoint_dir, step)
        print(f"Evaluating {name}", flush=True)
        results[name] = evaluate(
            model,
            identity,
            tokenizer,
            cases,
            qualitative,
            data_dir,
            comparison["max_tokens"],
            device,
        )
        atomic_json(output / f"{name}.json", results[name])
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pairs = list(zip(results["released"]["answers"], results["starter"]["answers"]))
    report = {
        "format": "speck_sft_pilot_comparison",
        "version": 1,
        "cases_sha256": file_sha256(case_path),
        "runner_sha256": file_sha256(__file__),
        "scorer_sha256": file_sha256(Path(__file__).with_name("instruct_eval.py")),
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "validation_data_directory": str(data_dir),
        "training_data_manifest_sha256": training_manifest_sha256,
        "dtype": "bfloat16",
        "decoding": "greedy",
        "models": results,
        "paired": {
            "released_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "starter_only_correct": sum(b["correct"] and not a["correct"] for a, b in pairs),
        },
        "scope": "Recipe pilot, not an official benchmark or isolated dataset-effect experiment. Starter validation can overlap the older release's upstream training sources. Inspect qualitative outputs separately.",
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                name: {
                    key: result[key]
                    for key in ("correct", "exact_format", "count", "starter_validation_loss")
                }
                for name, result in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
