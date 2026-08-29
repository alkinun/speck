"""Generate and score deterministic held-out SpeckGym evaluations."""

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, load_metadata, load_model, load_timing
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import SpeckForCausalLM
from speck.speckgym import resolve_training_phase
from speck.tokenizer import get_tokenizer
from speck.train import checkpoint_milestones

EVALUATION_FAMILIES = (
    "hierarchy",
    "retrieval",
    "binding",
    "state",
    "set_union",
    "composition",
)


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cases_fingerprint(cases):
    return _fingerprint(cases)


def _case_rng(seed, family, index):
    payload = f"{seed}\0heldout\0{family}\0{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _words(rng, count):
    alphabet = "abcdefghjkmnpqrstuvwxyz"
    words = []
    seen = set()
    while len(words) < count:
        word = "".join(rng.choice(alphabet) for _ in range(4))
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words


def _multiple_choice(family, index, prompt, correct, distractors, rng):
    choices = [correct]
    choices.extend(value for value in distractors if value != correct and value not in choices)
    if len(choices) < 4:
        raise ValueError(f"{family} case did not produce four unique choices")
    choices = choices[:4]
    rng.shuffle(choices)
    return {
        "id": f"{family}-{index:04d}",
        "family": family,
        "prompt": prompt,
        "choices": choices,
        "answer": choices.index(correct),
    }


def _hierarchy_case(index, rng):
    labels = _words(rng, 8)
    depth = rng.randint(4, 8)
    chain = labels[:depth]
    structure = " ".join(f"[{label}" for label in chain)
    structure += " " + " ".join(f"]{label}" for label in reversed(chain))
    query = rng.randrange(1, depth)
    prompt = f"Nested structure: {structure}\nDirect parent of {chain[query]}:"
    distractors = [label for label in chain if label != chain[query - 1]]
    return _multiple_choice("hierarchy", index, prompt, chain[query - 1], distractors, rng)


def _retrieval_case(index, rng):
    values = _words(rng, 10)
    position = rng.randrange(len(values))
    prompt = f"Ordered symbols: {' '.join(values)}\nSymbol at position {position + 1}:"
    return _multiple_choice(
        "retrieval",
        index,
        prompt,
        values[position],
        values[:position] + values[position + 1 :],
        rng,
    )


def _binding_case(index, rng):
    labels = _words(rng, 16)
    keys, values = labels[:8], labels[8:]
    pairs = list(zip(keys, values))
    rng.shuffle(pairs)
    query = rng.randrange(len(keys))
    prompt = "Bindings: " + "; ".join(f"{key}={value}" for key, value in pairs)
    prompt += f"\nValue bound to {keys[query]}:"
    return _multiple_choice("binding", index, prompt, values[query], values, rng)


def _state_case(index, rng):
    labels = _words(rng, 24)
    registers, values = labels[:5], labels[5:]
    state = dict(zip(registers, values[:5]))
    operations = [f"{register}={state[register]}" for register in registers]
    for value in values[5:15]:
        register = rng.choice(registers)
        state[register] = value
        operations.append(f"{register}={value}")
    query = rng.choice(registers)
    prompt = f"Apply updates in order: {'; '.join(operations)}\nFinal value of {query}:"
    return _multiple_choice("state", index, prompt, state[query], values, rng)


def _set_case(index, rng):
    labels = _words(rng, 12)
    left = labels[:4]
    right = labels[4:8]
    union = sorted((*left, *right))

    def render(values):
        return "{" + ",".join(sorted(values)) + "}"

    prompt = f"A={render(left)}; B={render(right)}\nA union B:"
    first = union.copy()
    first[0] = labels[8]
    second = union.copy()
    second[3] = labels[9]
    third = union.copy()
    third[1] = labels[10]
    third[6] = labels[11]
    return _multiple_choice(
        "set_union",
        index,
        prompt,
        render(union),
        (render(first), render(second), render(third)),
        rng,
    )


def _composition_case(index, rng):
    labels = _words(rng, 20)
    functions = labels[:5]
    values = labels[5:11]
    rules = [f"{functions[step]}({values[step]})={values[step + 1]}" for step in range(5)]
    rng.shuffle(rules)
    expression = values[0]
    for function in functions:
        expression = f"{function}({expression})"
    prompt = f"Rules: {'; '.join(rules)}\nEvaluate {expression}:"
    return _multiple_choice("composition", index, prompt, values[5], values, rng)


_CASE_GENERATORS = {
    "hierarchy": _hierarchy_case,
    "retrieval": _retrieval_case,
    "binding": _binding_case,
    "state": _state_case,
    "set_union": _set_case,
    "composition": _composition_case,
}


def generate_cases(seed, cases_per_family, families=EVALUATION_FAMILIES):
    """Generate a stable held-out set with domain-separated family seeds."""

    if not isinstance(cases_per_family, int) or cases_per_family < 1:
        raise ValueError("cases_per_family must be a positive integer")
    if tuple(families) != EVALUATION_FAMILIES:
        raise ValueError("SpeckGym evaluation families must use the checked order")
    cases = []
    for family in families:
        for index in range(cases_per_family):
            cases.append(_CASE_GENERATORS[family](index, _case_rng(seed, family, index)))
    return cases


@torch.no_grad()
def score_cases(model, tokenizer, cases, batch_size=4):
    """Score raw continuations by mean conditional token log probability."""

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("evaluation batch size must be a positive integer")
    encoded = []
    for case_index, case in enumerate(cases):
        prompt = tokenizer.encode(case["prompt"], bos=True)
        if not prompt:
            raise ValueError("evaluation prompt encoded to no tokens")
        for choice_index, choice in enumerate(case["choices"]):
            continuation = tokenizer.encode(" " + choice)
            if not continuation:
                raise ValueError("evaluation choice encoded to no tokens")
            tokens = prompt + continuation
            if len(tokens) > model.config.max_position_embeddings:
                raise ValueError("evaluation case exceeds the model context")
            encoded.append(
                {
                    "case": case_index,
                    "choice": choice_index,
                    "tokens": tokens,
                    "continuation_start": len(prompt),
                }
            )
    scores = [[None] * len(case["choices"]) for case in cases]
    device = next(model.parameters()).device
    for offset in range(0, len(encoded), batch_size):
        batch = encoded[offset : offset + batch_size]
        maximum = max(len(item["tokens"]) - 1 for item in batch)
        inputs = torch.full(
            (len(batch), maximum), tokenizer.eos_id, dtype=torch.long, device=device
        )
        for row, item in enumerate(batch):
            values = torch.tensor(item["tokens"][:-1], dtype=torch.long, device=device)
            inputs[row, : len(values)] = values
        log_probs = model(inputs).log_softmax(dim=-1)
        for row, item in enumerate(batch):
            tokens = item["tokens"]
            start = item["continuation_start"] - 1
            positions = torch.arange(start, len(tokens) - 1, device=device)
            targets = torch.tensor(tokens[item["continuation_start"] :], device=device)
            score = log_probs[row, positions, targets].mean().item()
            scores[item["case"]][item["choice"]] = score
    return scores


def summarize_scores(cases, scores):
    if len(cases) != len(scores):
        raise ValueError("case and score counts differ")
    totals = defaultdict(int)
    correct = defaultdict(int)
    margins = defaultdict(list)
    predictions = []
    for case, case_scores in zip(cases, scores):
        if any(score is None or not math.isfinite(score) for score in case_scores):
            raise ValueError("evaluation produced an invalid score")
        prediction = max(range(len(case_scores)), key=case_scores.__getitem__)
        family = case["family"]
        totals[family] += 1
        correct[family] += prediction == case["answer"]
        ordered = sorted(case_scores, reverse=True)
        margins[family].append(ordered[0] - ordered[1])
        predictions.append({"id": case["id"], "prediction": prediction, "scores": case_scores})
    metrics = {
        family: {
            "accuracy": correct[family] / totals[family],
            "mean_winning_margin": sum(margins[family]) / totals[family],
            "samples": totals[family],
        }
        for family in EVALUATION_FAMILIES
    }
    total = sum(totals.values())
    metrics["overall"] = {
        "accuracy": sum(correct.values()) / total,
        "chance_accuracy": 0.25,
        "samples": total,
    }
    return metrics, predictions


def _validate_language_checkpoint(metadata, train, step, requested_tokens):
    resolved = metadata.get("resolved", {})
    expected = {
        "run": train["run"],
        "batch_tokens": train["batch_tokens"],
        "train_tokens": train["train_tokens"],
        "global_token_offset": train["global_token_offset"],
        "checkpoint_tokens": train["checkpoint_tokens"],
        "initialization": train["initialization"],
    }
    changed = [key for key, value in expected.items() if resolved.get(key) != value]
    actual_tokens = train["global_token_offset"] + step * train["batch_tokens"]
    if changed:
        raise ValueError(f"checkpoint differs from the selected SpeckGym run: {', '.join(changed)}")
    if metadata.get("training_phase") != "language" or metadata.get("step") != step:
        raise ValueError("checkpoint is not the selected SpeckGym language phase")
    if metadata.get("milestone_tokens") != requested_tokens:
        raise ValueError("checkpoint metadata does not match the requested milestone")
    if metadata.get("global_tokens") != actual_tokens:
        raise ValueError("checkpoint has an invalid global token position")
    if metadata.get("validation_global_tokens") != actual_tokens:
        raise ValueError("checkpoint does not contain fresh milestone validation")


def resolve_language_checkpoint(suite, run, requested_tokens, cache_dir=None):
    """Resolve one requested global-token milestone to its native language checkpoint."""

    if requested_tokens not in suite["checkpoint_tokens"]:
        raise ValueError("tokens must select a configured SpeckGym checkpoint")
    base_configs = load_experiment(suite["base_experiment"], "data", "tokenizer", "model", "train")
    configs = resolve_training_phase(suite, base_configs, run, "language", cache_dir)
    train = configs["train"]
    steps = math.ceil(train["train_tokens"] / train["batch_tokens"])
    milestones = checkpoint_milestones(
        train["checkpoint_tokens"],
        train["batch_tokens"],
        train["global_token_offset"],
        steps,
    )
    matches = [step for step, token in milestones.items() if token == requested_tokens]
    if len(matches) != 1:
        raise ValueError("requested checkpoint does not resolve uniquely")
    step = matches[0]
    directory = Path(train["output_dir"])
    metadata = load_metadata(directory, step)
    _validate_language_checkpoint(metadata, train, step, requested_tokens)
    return configs, directory, step, metadata


def evaluate_procedural_checkpoint(
    suite,
    run,
    requested_tokens,
    *,
    device=None,
    batch_size=None,
    output_dir=None,
    cache_dir=None,
):
    configs, checkpoint_dir, step, metadata = resolve_language_checkpoint(
        suite, run, requested_tokens, cache_dir
    )
    tokenizer = get_tokenizer(**configs["tokenizer"])
    config = ArchitectureConfig.from_dict(metadata["config"])
    if (tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id) != (
        config.vocab_size,
        config.bos_token_id,
        config.eos_token_id,
    ):
        raise ValueError("checkpoint and evaluation tokenizer do not match")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpeckForCausalLM(config).to(device)
    model.load_state_dict(load_model(checkpoint_dir, step, device))
    model.eval()
    evaluation = suite["evaluation"]
    cases = generate_cases(
        evaluation["seed"], evaluation["cases_per_family"], evaluation["families"]
    )
    scores = score_cases(model, tokenizer, cases, batch_size or evaluation["batch_size"])
    metrics, predictions = summarize_scores(cases, scores)
    report = {
        "format_version": 1,
        "run": run,
        "requested_tokens": requested_tokens,
        "actual_tokens": metadata["global_tokens"],
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "cases": {
            "seed": evaluation["seed"],
            "cases_per_family": evaluation["cases_per_family"],
            "families": evaluation["families"],
            "sha256": cases_fingerprint(cases),
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    output_dir = Path(
        output_dir or Path(base_dir()) / "evaluations" / "SpeckGym-v0" / run / str(requested_tokens)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "procedural.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def training_metrics(suite, run, requested_tokens, cache_dir=None):
    """Aggregate warm-up and language timing at one common global-token milestone."""

    _, directory, step, metadata = resolve_language_checkpoint(
        suite, run, requested_tokens, cache_dir
    )
    language_timing = load_timing(directory, step) or metadata.get("timing", {})
    return aggregate_training_metrics(metadata, language_timing, run)


def aggregate_training_metrics(metadata, language_timing, run):
    """Combine embedded phase timing without consulting the warm-up checkpoint."""

    language_tokens = metadata["step"] * metadata["resolved"]["batch_tokens"]
    phases = [
        {
            "name": "language",
            "checkpoint_step": metadata["step"],
            "tokens": language_tokens,
            **language_timing,
        }
    ]
    if run != "A":
        initialization = metadata.get("initialization")
        if not isinstance(initialization, dict) or not isinstance(
            initialization.get("source_timing"), dict
        ):
            raise ValueError("language checkpoint does not embed warm-up timing provenance")
        phases.insert(
            0,
            {
                "name": "procedural_warmup",
                "checkpoint_step": initialization["step"],
                "tokens": initialization["source_tokens"],
                **initialization["source_timing"],
            },
        )
    optimizer_seconds = sum(phase.get("optimizer_seconds", 0.0) for phase in phases)
    active_seconds = sum(phase.get("active_seconds", 0.0) for phase in phases)
    evaluation_seconds = sum(phase.get("evaluation_seconds", 0.0) for phase in phases)
    checkpoint_seconds = sum(phase.get("checkpoint_seconds", 0.0) for phase in phases)
    return {
        "phases": phases,
        "optimizer_seconds": optimizer_seconds,
        "active_seconds": active_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "average_tokens_per_second": (
            metadata["global_tokens"] / optimizer_seconds if optimizer_seconds else None
        ),
    }
