"""Adapt a checkpoint on deterministic structured retrieval and measure held-out specificity."""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.infer import load_checkpoint_model
from scripts.structured_retrieval_eval import TASKS, build_case, parse_tasks
from speck.checkpoint import checkpoint_identity, latest, save
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.long_context import (
    ANSWER_SETS,
    RETRIEVAL_TEMPLATES,
    binomial_tail_probability,
)
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, set_optimizer_lr

TRAIN_SEED_OFFSET = 1_000_000
VALIDATION_SEED_OFFSET = 2_000_000


def parse_choice_list(value, choices, name):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(item not in choices for item in values):
        raise ValueError(f"{name} must be unique values from {', '.join(choices)}")
    return values


def parse_templates(value):
    return parse_choice_list(value, RETRIEVAL_TEMPLATES, "templates")


def parse_answer_sets(value):
    return parse_choice_list(value, tuple(ANSWER_SETS), "answer sets")


def parse_record_counts(value):
    try:
        counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("record counts must be unique integers in [2, 10]") from error
    if not counts or len(set(counts)) != len(counts) or any(not 2 <= count <= 10 for count in counts):
        raise ValueError("record counts must be unique integers in [2, 10]")
    return counts


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tasks", type=parse_tasks, default=TASKS)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--chains", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--min-lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed-offset", type=int, default=TRAIN_SEED_OFFSET)
    parser.add_argument("--validation-seed-offset", type=int, default=VALIDATION_SEED_OFFSET)
    parser.add_argument("--train-templates", type=parse_templates, default=("archive",))
    parser.add_argument("--validation-templates", type=parse_templates, default=("archive",))
    parser.add_argument("--train-answer-sets", type=parse_answer_sets, default=("letters",))
    parser.add_argument("--validation-answer-sets", type=parse_answer_sets, default=("letters",))
    parser.add_argument("--train-record-counts", type=parse_record_counts, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--replay-data-experiment", type=Path, default=None)
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_revision():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_supervised_batch(
    tokenizer,
    tasks,
    sequence_length,
    batch_size,
    first_sample,
    records,
    chains,
    device,
    templates=("archive",),
    answer_sets=("letters",),
    record_counts=None,
):
    record_counts = tuple(record_counts or (records,))
    cases = []
    for offset in range(batch_size):
        index = first_sample + offset
        task = tasks[index % len(tasks)]
        template = templates[(index // len(tasks)) % len(templates)]
        answer_set = answer_sets[
            (index // (len(tasks) * len(templates))) % len(answer_sets)
        ]
        record_count = record_counts[
            (index // (len(tasks) * len(templates) * len(answer_sets))) % len(record_counts)
        ]
        case = build_case(
            task,
            tokenizer,
            sequence_length + 1,
            index,
            record_count,
            chains,
            template=template,
            answer_set=answer_set,
        )
        inputs = case["prompt_tokens"] + case["answer_tokens"][:-1]
        if len(inputs) != sequence_length:
            raise RuntimeError("structured adaptation case has invalid training geometry")
        case["input_tokens"] = inputs
        cases.append(case)
    inputs = torch.tensor([case["input_tokens"] for case in cases], device=device)
    targets = torch.full_like(inputs, -100)
    for row, case in enumerate(cases):
        start = case["prompt_length"] - 1
        stop = start + len(case["answer_tokens"])
        targets[row, start:stop] = torch.tensor(case["answer_tokens"], device=device)
    return inputs, targets, cases


def candidate_shift(reference, changed, from_indices, to_indices):
    rows = torch.arange(reference.size(0), device=reference.device)
    reference_preference = reference[rows, from_indices] - reference[rows, to_indices]
    changed_preference = changed[rows, to_indices] - changed[rows, from_indices]
    return (reference_preference + changed_preference) / 2


def replay_microsteps(accumulation, fraction):
    """Choose evenly spaced replay microsteps for an exactly representable fraction."""

    accumulation = positive_integer(accumulation, "accumulation")
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("replay fraction must be in [0, 1)")
    replay_count = round(accumulation * fraction)
    if not math.isclose(replay_count / accumulation, fraction):
        raise ValueError("replay fraction must be exactly representable by accumulation")
    if not replay_count:
        return ()
    return tuple((index + 1) * accumulation // replay_count - 1 for index in range(replay_count))


@torch.inference_mode()
def validate(model, tokenizer, settings, device):
    model.eval()
    metrics = {}
    for task_index, task in enumerate(settings["tasks"]):
        for template_index, template in enumerate(settings["validation_templates"]):
            for answer_set_index, answer_set in enumerate(settings["validation_answer_sets"]):
                totals = {
                    "samples": 0,
                    "answer_tokens": 0,
                    "exact": 0,
                    "token_correct": 0,
                    "candidate": 0,
                    "target_direction": 0,
                    "specificity_direction": 0,
                    "target_score": 0.0,
                    "distractor_score": 0.0,
                    "specificity_score": 0.0,
                }
                for start in range(0, settings["validation_samples"], settings["batch_size"]):
                    count = min(
                        settings["batch_size"], settings["validation_samples"] - start
                    )
                    cases = []
                    counterfactuals = []
                    distractors = []
                    for offset in range(count):
                        seed = (
                            settings["validation_seed_offset"]
                            + task_index * 100_000
                            + template_index * 10_000_000
                            + answer_set_index * 1_000_000
                            + start
                            + offset
                        )
                        case = build_case(
                            task,
                            tokenizer,
                            settings["sequence_length"] + 1,
                            seed,
                            settings["records"],
                            settings["chains"],
                            template=template,
                            answer_set=answer_set,
                        )
                        counterfactual = build_case(
                            task,
                            tokenizer,
                            settings["sequence_length"] + 1,
                            seed,
                            settings["records"],
                            settings["chains"],
                            answer_offset=1,
                            template=template,
                            answer_set=answer_set,
                        )
                        distractor_index = (case["query_index"] + 1) % (
                            case.get("records") or case["chains"]
                        )
                        distractor = build_case(
                            task,
                            tokenizer,
                            settings["sequence_length"] + 1,
                            seed,
                            settings["records"],
                            settings["chains"],
                            answer_offset=1,
                            mutation_index=distractor_index,
                            template=template,
                            answer_set=answer_set,
                        )
                        cases.append(case)
                        counterfactuals.append(counterfactual)
                        distractors.append(distractor)

                    def logits(values, answer_prefix=0):
                        prompts = torch.tensor(
                            [
                                case["prompt_tokens"] + case["answer_tokens"][:answer_prefix]
                                for case in values
                            ],
                            device=device,
                        )
                        return model(prompts, last_token_only=True)[:, -1]

                    factual_logits = logits(cases)
                    counterfactual_logits = logits(counterfactuals)
                    distractor_logits = logits(distractors)
                    candidates = torch.tensor(cases[0]["candidate_token_ids"], device=device)
                    factual_candidates = factual_logits[:, candidates].log_softmax(dim=-1)
                    counterfactual_candidates = counterfactual_logits[:, candidates].log_softmax(
                        dim=-1
                    )
                    distractor_candidates = distractor_logits[:, candidates].log_softmax(dim=-1)
                    answer_indices = torch.tensor(
                        [case["answer_index"] for case in cases], device=device
                    )
                    changed_indices = torch.tensor(
                        [case["answer_index"] for case in counterfactuals], device=device
                    )
                    distractor_from = torch.tensor(
                        [case["mutation_from_index"] for case in distractors], device=device
                    )
                    distractor_to = torch.tensor(
                        [case["mutation_to_index"] for case in distractors], device=device
                    )
                    target_score = candidate_shift(
                        factual_candidates,
                        counterfactual_candidates,
                        answer_indices,
                        changed_indices,
                    )
                    distractor_score = candidate_shift(
                        factual_candidates,
                        distractor_candidates,
                        distractor_from,
                        distractor_to,
                    )
                    specificity = target_score - distractor_score
                    answer_lengths = {len(case["answer_tokens"]) for case in cases}
                    if len(answer_lengths) != 1:
                        raise RuntimeError("validation answer lengths changed within a batch")
                    answer_length = answer_lengths.pop()
                    exact = torch.ones(count, dtype=torch.bool, device=device)
                    for answer_position in range(answer_length):
                        position_logits = (
                            factual_logits if answer_position == 0 else logits(cases, answer_position)
                        )
                        expected = torch.tensor(
                            [case["answer_tokens"][answer_position] for case in cases],
                            device=device,
                        )
                        correct = position_logits.argmax(dim=-1) == expected
                        exact &= correct
                        totals["token_correct"] += int(correct.sum().item())
                    totals["samples"] += count
                    totals["answer_tokens"] += count * answer_length
                    totals["exact"] += int(exact.sum().item())
                    totals["candidate"] += int(
                        (factual_candidates.argmax(dim=-1) == answer_indices).sum().item()
                    )
                    totals["target_direction"] += int((target_score > 0).sum().item())
                    totals["specificity_direction"] += int((specificity > 0).sum().item())
                    totals["target_score"] += target_score.sum().item()
                    totals["distractor_score"] += distractor_score.sum().item()
                    totals["specificity_score"] += specificity.sum().item()
                samples = totals["samples"]
                key = (
                    task
                    if settings["validation_templates"] == ("archive",)
                    and settings["validation_answer_sets"] == ("letters",)
                    else f"{task}/{template}/{answer_set}"
                )
                metrics[key] = {
                    "task": task,
                    "template": template,
                    "answer_set": answer_set,
                    "samples": samples,
                    "exact_match": totals["exact"] / samples,
                    "token_accuracy": totals["token_correct"] / totals["answer_tokens"],
                    "candidate_accuracy": totals["candidate"] / samples,
                    "candidate_p_value": binomial_tail_probability(
                        totals["candidate"], samples, 0.1
                    ),
                    "target_direction_accuracy": totals["target_direction"] / samples,
                    "target_direction_p_value": binomial_tail_probability(
                        totals["target_direction"], samples, 0.5
                    ),
                    "association_specificity_accuracy": (
                        totals["specificity_direction"] / samples
                    ),
                    "association_specificity_p_value": binomial_tail_probability(
                        totals["specificity_direction"], samples, 0.5
                    ),
                    "target_change_score": totals["target_score"] / samples,
                    "distractor_change_score": totals["distractor_score"] / samples,
                    "association_specificity_score": totals["specificity_score"] / samples,
                }
    return metrics


def validate_settings(args):
    settings = {
        "tasks": tuple(args.tasks),
        "sequence_length": positive_integer(args.sequence_length, "sequence length"),
        "steps": positive_integer(args.steps, "steps"),
        "batch_size": positive_integer(args.batch_size, "batch size"),
        "accumulation": positive_integer(args.accumulation, "accumulation"),
        "validation_samples": positive_integer(args.validation_samples, "validation samples"),
        "eval_every": positive_integer(args.eval_every, "evaluation interval"),
        "records": positive_integer(args.records, "records"),
        "chains": positive_integer(args.chains, "chains"),
        "warmup_steps": positive_integer(args.warmup_steps, "warmup steps"),
        "seed": args.seed,
        "train_seed_offset": getattr(args, "train_seed_offset", TRAIN_SEED_OFFSET),
        "validation_seed_offset": getattr(
            args, "validation_seed_offset", VALIDATION_SEED_OFFSET
        ),
        "train_templates": tuple(getattr(args, "train_templates", ("archive",))),
        "validation_templates": tuple(getattr(args, "validation_templates", ("archive",))),
        "train_answer_sets": tuple(getattr(args, "train_answer_sets", ("letters",))),
        "validation_answer_sets": tuple(
            getattr(args, "validation_answer_sets", ("letters",))
        ),
        "train_record_counts": tuple(
            getattr(args, "train_record_counts", None) or (args.records,)
        ),
        "replay_fraction": getattr(args, "replay_fraction", 0.0),
    }
    for name in ("train_seed_offset", "validation_seed_offset"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name.replace('_', ' ')} must be a non-negative integer")
    for name, choices, label in (
        ("train_templates", RETRIEVAL_TEMPLATES, "train templates"),
        ("validation_templates", RETRIEVAL_TEMPLATES, "validation templates"),
        ("train_answer_sets", tuple(ANSWER_SETS), "train answer sets"),
        ("validation_answer_sets", tuple(ANSWER_SETS), "validation answer sets"),
    ):
        settings[name] = parse_choice_list(",".join(settings[name]), choices, label)
    settings["train_record_counts"] = parse_record_counts(
        ",".join(str(count) for count in settings["train_record_counts"])
    )
    for name in ("lr", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        settings[name] = value
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative and finite")
    if not math.isfinite(args.min_lr) or not 0 <= args.min_lr <= 1:
        raise ValueError("minimum LR multiplier must be in [0, 1]")
    settings.update(
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        optimizer=args.optimizer,
    )
    settings["replay_microsteps"] = replay_microsteps(
        settings["accumulation"], settings["replay_fraction"]
    )
    return settings


def run(args):
    settings = validate_settings(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"retrieval adaptation output already exists: {output_dir}")
    configs = load_experiment(args.experiment, "tokenizer")
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(settings["seed"])
    model, parent_metadata = load_checkpoint_model(checkpoint_dir, step, device)
    if settings["sequence_length"] + 1 > model.config.max_position_embeddings:
        raise ValueError("adaptation sequence exceeds the model context")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    replay_loader = None
    replay_manifest = None
    replay_data_dir = None
    if bool(settings["replay_microsteps"]) != (args.replay_data_experiment is not None):
        raise ValueError("positive replay requires exactly one replay data experiment")
    if args.replay_data_experiment is not None:
        replay_configs = load_experiment(args.replay_data_experiment, "data", "tokenizer")
        if replay_configs["tokenizer"] != configs["tokenizer"]:
            raise ValueError("replay tokenizer does not match the adapted model")
        replay_data_dir = resolve_data_dir(
            replay_configs["data"].get("output_dir"),
            replay_configs["data"].get("output_name"),
        )
        replay_manifest = load_manifest(replay_data_dir)
        replay_loader = packed_loader(
            tokenizer,
            settings["batch_size"],
            settings["sequence_length"],
            "train",
            device=device,
            data_dir=replay_data_dir,
        )
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(settings["lr"], settings["weight_decay"], settings["optimizer"])
    train_model = (
        model
        if args.no_compile
        else torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
    )
    history = []
    model.eval()
    history.append({"step": 0, "validation": validate(model, tokenizer, settings, device)})
    model.train()
    started = time.perf_counter()
    retrieval_sample = settings["train_seed_offset"]
    retrieval_examples_seen = 0
    retrieval_supervised_tokens_seen = 0
    retrieval_prompt_tokens_seen = 0
    retrieval_examples_by_condition = {}
    replay_tokens_seen = 0
    for step_index in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        retrieval_loss_sum = torch.zeros((), device=device)
        replay_loss_sum = torch.zeros((), device=device)
        retrieval_microbatches = 0
        replay_microbatches = 0
        for micro_step in range(settings["accumulation"]):
            if micro_step in settings["replay_microsteps"]:
                if replay_loader is None:
                    raise RuntimeError("replay schedule has no data loader")
                inputs, targets, _ = next(replay_loader)
                loss = train_model(inputs, targets)
                replay_loss_sum += loss.detach()
                replay_microbatches += 1
                replay_tokens_seen += settings["batch_size"] * settings["sequence_length"]
            else:
                inputs, targets, cases = build_supervised_batch(
                    tokenizer,
                    settings["tasks"],
                    settings["sequence_length"],
                    settings["batch_size"],
                    retrieval_sample,
                    settings["records"],
                    settings["chains"],
                    device,
                    settings["train_templates"],
                    settings["train_answer_sets"],
                    settings["train_record_counts"],
                )
                retrieval_sample += settings["batch_size"]
                retrieval_examples_seen += settings["batch_size"]
                retrieval_supervised_tokens_seen += sum(
                    len(case["answer_tokens"]) for case in cases
                )
                retrieval_prompt_tokens_seen += sum(case["prompt_length"] for case in cases)
                for case in cases:
                    condition = f"{case['task']}/{case['template']}/{case['answer_set']}"
                    if len(settings["train_record_counts"]) > 1:
                        condition += f"/records_{case['records']}"
                    retrieval_examples_by_condition[condition] = (
                        retrieval_examples_by_condition.get(condition, 0) + 1
                    )
                loss = train_model(inputs, targets, loss_reduction="sum") / settings["batch_size"]
                retrieval_loss_sum += loss.detach()
                retrieval_microbatches += 1
            (loss / settings["accumulation"]).backward()
            loss_sum += loss.detach()
        scale = lr_scale(
            step_index,
            settings["steps"],
            settings["warmup_steps"],
            settings["min_lr"],
        )
        set_optimizer_lr(optimizer, settings["lr"] * scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"])
        if not torch.isfinite(loss_sum) or not torch.isfinite(grad_norm):
            raise FloatingPointError("retrieval adaptation produced non-finite optimization state")
        optimizer.step()
        completed = step_index + 1
        if completed == 1 or completed % 10 == 0:
            print(
                f"step {completed}/{settings['steps']} | "
                f"loss {loss_sum.item() / settings['accumulation']:.5f} | "
                f"retrieval {retrieval_loss_sum.item() / max(1, retrieval_microbatches):.5f} | "
                f"replay {replay_loss_sum.item() / max(1, replay_microbatches):.5f} | "
                f"grad {float(grad_norm):.3f}"
            )
        if completed % settings["eval_every"] == 0 or completed == settings["steps"]:
            validation = validate(model, tokenizer, settings, device)
            history.append({"step": completed, "validation": validation})
            print(
                f"step {completed} validation | "
                + " | ".join(
                    f"{task} choice={values['candidate_accuracy']:.3f} "
                    f"specificity={values['association_specificity_accuracy']:.3f}"
                    for task, values in validation.items()
                )
            )
            model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metadata = {
        "format_version": 1,
        "training_phase": "structured_retrieval_adaptation",
        "step": settings["steps"],
        "config": model.config.settings(),
        "parent_checkpoint": checkpoint_identity(checkpoint_dir, step),
        "parent_global_tokens": parent_metadata.get("global_tokens"),
        "settings": {**settings, "tasks": list(settings["tasks"])},
        "history": history,
        "training_seconds": training_seconds,
        "git_revision": git_revision(),
        "replay_data": (
            {
                "experiment": str(args.replay_data_experiment.expanduser().resolve()),
                "directory": str(replay_data_dir.expanduser().resolve()),
                "manifest": manifest_fingerprint(replay_manifest),
            }
            if replay_manifest is not None
            else None
        ),
    }
    save(
        output_dir,
        settings["steps"],
        model.state_dict(),
        optimizer.state_dict(),
        metadata,
    )
    report = {
        "format": "speck_structured_retrieval_adaptation",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(args.experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(output_dir, settings["steps"]),
        "parent_checkpoint": metadata["parent_checkpoint"],
        "settings": metadata["settings"],
        "history": history,
        "training_seconds": training_seconds,
        "tokens_seen": (
            settings["steps"]
            * settings["batch_size"]
            * settings["accumulation"]
            * settings["sequence_length"]
        ),
        "supervised_tokens_seen": retrieval_supervised_tokens_seen,
        "retrieval_examples_seen": retrieval_examples_seen,
        "retrieval_examples_by_condition": dict(sorted(retrieval_examples_by_condition.items())),
        "retrieval_input_tokens_seen": retrieval_examples_seen * settings["sequence_length"],
        "retrieval_prompt_tokens_seen": retrieval_prompt_tokens_seen,
        "replay_tokens_seen": replay_tokens_seen,
        "replay_data": metadata["replay_data"],
        "device": str(device),
        "torch_version": torch.__version__,
        "git_revision": metadata["git_revision"],
    }
    atomic_json(args.report, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    final = report["history"][-1]
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
