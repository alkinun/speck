"""Compare local checkpoints for instruction-tuned Speck models on 15 fixed questions."""

import argparse
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.chat import ChatTokenizer
from speck.checkpoint import completed_steps, load_model
from speck.generation import generate_tokens
from speck.model import SpeckForCausalLM
from speck.tokenizer import Tokenizer

SCORING_VERSION = 2
NUMBER_PATTERN = re.compile(
    r"(?<![\w.,+\-/])[+-]?[ \t]*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?(?!\w|[.,]\d|/)"
)

QUESTIONS = (
    {
        "category": "knowledge",
        "prompt": "What is the capital of France? Answer with only the city name.",
        "accepted": ("Paris",),
    },
    {
        "category": "knowledge",
        "prompt": "Which planet is known as the Red Planet? Answer with only the planet name.",
        "accepted": ("Mars",),
    },
    {
        "category": "knowledge",
        "prompt": "Who wrote the novel 1984? Answer with only the author's name.",
        "accepted": ("George Orwell", "Orwell"),
    },
    {
        "category": "knowledge",
        "prompt": "What is the chemical symbol for gold? Answer with only the symbol.",
        "accepted": ("Au",),
    },
    {
        "category": "knowledge",
        "prompt": "What is the largest ocean on Earth? Answer with only its name.",
        "accepted": ("Pacific", "Pacific Ocean"),
    },
    {
        "category": "reasoning",
        "prompt": "Calculate 17 + 28. Answer with only the number.",
        "accepted": ("45",),
        "scoring": "final_number",
    },
    {
        "category": "reasoning",
        "prompt": "Calculate 12 multiplied by 7. Answer with only the number.",
        "accepted": ("84",),
        "scoring": "final_number",
    },
    {
        "category": "reasoning",
        "prompt": (
            "Maya has 15 apples and gives away 6. How many apples remain? "
            "Answer with only the number."
        ),
        "accepted": ("9",),
        "scoring": "final_number",
    },
    {
        "category": "reasoning",
        "prompt": "What number comes next: 3, 6, 12, 24? Answer with only the number.",
        "accepted": ("48",),
        "scoring": "final_number",
    },
    {
        "category": "reasoning",
        "prompt": (
            "All tulips are flowers, and all flowers are plants. Are all tulips plants? "
            "Answer only yes or no."
        ),
        "accepted": ("yes",),
    },
    {
        "category": "instruction",
        "prompt": "Give the opposite of the word 'scarce'. Answer with one word only.",
        "accepted": ("abundant", "plentiful"),
    },
    {
        "category": "instruction",
        "prompt": "Translate 'good morning' into Spanish. Answer with only the translation.",
        "accepted": ("buenos dias",),
    },
    {
        "category": "instruction",
        "prompt": (
            "Put these words in alphabetical order: pear, apple, orange. "
            "Answer with only the ordered words."
        ),
        "accepted": ("apple orange pear",),
    },
    {
        "category": "instruction",
        "prompt": "In Python, what does len(['a', 'b', 'c']) return? Answer with only the number.",
        "accepted": ("3",),
        "scoring": "final_number",
    },
    {
        "category": "instruction",
        "prompt": "Reply with exactly these two words and nothing else: blue triangle",
        "accepted": ("blue triangle",),
        "scoring": "exact",
    },
)

MODEL_STEPS = {
    "Speck1-140M-Instruct": 4_835,
    "Speck1.1-140M-Instruct": 8_534,
    "Speck1.1-140M-Instruct-2ep": 17_068,
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path.home() / ".cache/speck/checkpoints",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("instruct_eval_results.json"))
    return parser.parse_args()


def normalize(value):
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _numeric_text(value):
    return unicodedata.normalize("NFKC", value).replace("−", "-").strip()


def _number_value(value):
    return Decimal("".join(value.split()).replace(",", ""))


def _score_final_number(answer, accepted):
    text = _numeric_text(answer)
    expected = set()
    for value in accepted:
        value = _numeric_text(value)
        if NUMBER_PATTERN.fullmatch(value) is None:
            raise ValueError(f"invalid accepted numeric answer: {value!r}")
        expected.add(_number_value(value))
    numbers = NUMBER_PATTERN.findall(text)
    if not numbers:
        return False, False
    try:
        correct = _number_value(numbers[-1]) in expected
    except InvalidOperation:
        return False, False
    exact = correct and NUMBER_PATTERN.fullmatch(text) is not None
    return correct, exact


def score(answer, accepted, scoring="contains"):
    if scoring == "final_number":
        return _score_final_number(answer, accepted)
    if scoring not in {"contains", "exact"}:
        raise ValueError(f"unsupported instruction scoring mode: {scoring}")
    normalized = normalize(answer)
    expected = tuple(normalize(value) for value in accepted)
    exact = normalized in expected
    if scoring == "exact":
        correct = exact
    else:
        correct = exact or any(
            re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", normalized)
            for value in expected
        )
    return correct, exact


def generate(model, tokenizer, prompt, max_tokens, device):
    tokens, _ = tokenizer.encode_messages(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    generated = generate_tokens(
        model, tokens, max_tokens=max_tokens, eos_token_id=tokenizer.eos_id, device=device
    )
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


def evaluate(name, checkpoint_dir, step, max_tokens, device):
    if step not in completed_steps(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint {step} is incomplete in {checkpoint_dir}")
    metadata_path = checkpoint_dir / f"metadata_{step:06d}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tokenizer = ChatTokenizer(Tokenizer(checkpoint_dir / "tokenizer/tokenizer.model"))
    if metadata.get("resolved", {}).get("tokenizer") != tokenizer.metadata():
        raise ValueError(f"checkpoint and tokenizer do not match for {name}")

    model = SpeckForCausalLM(ArchitectureConfig.from_dict(metadata["config"]))
    model.load_state_dict(load_model(checkpoint_dir, step, "cpu"))
    model.to(device).eval()

    answers = []
    for index, question in enumerate(QUESTIONS, start=1):
        answer, generated_tokens = generate(
            model, tokenizer, question["prompt"], max_tokens, device
        )
        correct, exact = score(answer, question["accepted"], question.get("scoring", "contains"))
        answers.append(
            {
                "number": index,
                **question,
                "answer": answer,
                "generated_tokens": generated_tokens,
                "correct": correct,
                "exact_format": exact,
            }
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    categories = {
        category: {
            "correct": sum(
                answer["correct"] for answer in answers if answer["category"] == category
            ),
            "questions": sum(answer["category"] == category for answer in answers),
        }
        for category in ("knowledge", "reasoning", "instruction")
    }
    return {
        "name": name,
        "source_run": metadata["resolved"]["run"],
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "step": step,
        "correct": sum(answer["correct"] for answer in answers),
        "exact_format": sum(answer["exact_format"] for answer in answers),
        "questions": len(answers),
        "categories": categories,
        "answers": answers,
    }


def main():
    args = arguments()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    device = torch.device(args.device)
    results = {
        "method": {
            "scoring_version": SCORING_VERSION,
            "decoding": "greedy",
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "primary_metric": "correct answers out of 15",
            "tie_breaker": "exact-format answers out of 15",
            "device": str(device),
        },
        "models": [
            evaluate(name, args.checkpoints_root / name, step, args.max_tokens, device)
            for name, step in MODEL_STEPS.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results["models"]:
        print(
            f"{result['name']}: {result['correct']}/{result['questions']} correct, "
            f"{result['exact_format']}/{result['questions']} exact-format answers"
        )
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
