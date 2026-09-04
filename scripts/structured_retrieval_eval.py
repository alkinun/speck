"""Evaluate multi-key and two-hop retrieval with exact and counterfactual metrics."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.infer import load_checkpoint_model
from scripts.long_context_eval import positional_regime
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.long_context import (
    ANSWER_SETS,
    RETRIEVAL_TEMPLATES,
    add_counterfactual_metrics,
    aggregate_results,
    build_multi_key_case,
    build_symbolic_two_hop_case,
    build_two_hop_case,
    candidate_shift_score,
    evaluate_case,
    parse_lengths,
)
from speck.tokenizer import get_tokenizer

PRIMARY_TASKS = ("multi_key", "two_hop")
SYMBOLIC_TASK_MODES = {
    "two_hop_route": "route",
    "two_hop_payload": "payload",
    "two_hop_symbolic": "compose",
}
TASKS = PRIMARY_TASKS + tuple(SYMBOLIC_TASK_MODES)
TWO_HOP_DEPTHS = ((0.1, 0.5), (0.1, 0.9), (0.5, 0.9))


def parse_tasks(value):
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be unique values from {', '.join(TASKS)}")
    return tasks


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tasks", type=parse_tasks, default=PRIMARY_TASKS)
    parser.add_argument("--lengths", type=parse_lengths, default=(4_096, 32_768, 131_072))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--chains", type=int, default=6)
    parser.add_argument("--template", choices=RETRIEVAL_TEMPLATES, default="archive")
    parser.add_argument("--answer-set", choices=tuple(ANSWER_SETS), default="letters")
    parser.add_argument("--response-cue", choices=("native", "answer"), default="native")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_case(
    task,
    tokenizer,
    length,
    seed,
    records,
    chains,
    answer_offset=0,
    mutation_index=None,
    template="archive",
    answer_set="letters",
    response_cue="native",
):
    if task == "multi_key":
        return build_multi_key_case(
            tokenizer,
            length,
            seed,
            depth=(0.1, 0.5, 0.9)[seed % 3],
            records=records,
            answer_offset=answer_offset,
            mutation_index=mutation_index,
            template=template,
            answer_set=answer_set,
            response_cue=response_cue,
        )
    first_depth, second_depth = TWO_HOP_DEPTHS[seed % len(TWO_HOP_DEPTHS)]
    if task == "two_hop":
        return build_two_hop_case(
            tokenizer,
            length,
            seed,
            first_depth,
            second_depth,
            chains=chains,
            answer_offset=answer_offset,
            mutation_index=mutation_index,
            template=template,
            answer_set=answer_set,
            response_cue=response_cue,
        )
    try:
        mode = SYMBOLIC_TASK_MODES[task]
    except KeyError as error:
        raise ValueError(f"unsupported structured retrieval task: {task}") from error
    return build_symbolic_two_hop_case(
        tokenizer,
        length,
        seed,
        first_depth,
        second_depth,
        chains=chains,
        answer_offset=answer_offset,
        mutation_index=mutation_index,
        mode=mode,
        template=template,
        answer_set=answer_set,
        response_cue=response_cue,
    )


def run(args):
    samples = positive_integer(args.samples, "samples")
    records = positive_integer(args.records, "records")
    chains = positive_integer(args.chains, "chains")
    configs = load_experiment(args.experiment, "tokenizer", "train")
    checkpoint_dir = args.checkpoint_dir or Path(
        configs["train"].get("output_dir")
        or Path(base_dir()) / "checkpoints" / configs["train"]["run"]
    )
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    device = torch.device(args.device)
    model, _ = load_checkpoint_model(checkpoint_dir, step, device)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    if max(args.lengths) > model.config.max_position_embeddings:
        raise ValueError("evaluation length exceeds the model's allocated context")
    results = []
    task_summaries = {}
    for task in args.tasks:
        task_results = []
        for length in args.lengths:
            for sample in range(samples):
                case = build_case(
                    task,
                    tokenizer,
                    length,
                    sample,
                    records,
                    chains,
                    template=args.template,
                    answer_set=args.answer_set,
                    response_cue=args.response_cue,
                )
                counterfactual_case = build_case(
                    task,
                    tokenizer,
                    length,
                    sample,
                    records,
                    chains,
                    answer_offset=1,
                    template=args.template,
                    answer_set=args.answer_set,
                    response_cue=args.response_cue,
                )
                distractor_index = (case["query_index"] + 1) % (
                    case.get("records") or case["chains"]
                )
                distractor_case = build_case(
                    task,
                    tokenizer,
                    length,
                    sample,
                    records,
                    chains,
                    answer_offset=1,
                    mutation_index=distractor_index,
                    template=args.template,
                    answer_set=args.answer_set,
                    response_cue=args.response_cue,
                )
                factual = evaluate_case(model, case, device=device, kv_cache_dtype=torch.bfloat16)
                counterfactual = evaluate_case(
                    model,
                    counterfactual_case,
                    device=device,
                    kv_cache_dtype=torch.bfloat16,
                )
                result = add_counterfactual_metrics(
                    factual, counterfactual, case, counterfactual_case
                )
                distractor = evaluate_case(
                    model,
                    distractor_case,
                    device=device,
                    kv_cache_dtype=torch.bfloat16,
                )
                distractor_change_score = candidate_shift_score(
                    result,
                    distractor,
                    distractor_case["mutation_from_index"],
                    distractor_case["mutation_to_index"],
                )
                association_specificity_score = (
                    result["contrastive_retrieval_score"] - distractor_change_score
                )
                result.update(
                    fact_positions=case["fact_positions"],
                    template=case["template"],
                    answer_set=case["answer_set"],
                    response_cue=case["response_cue"],
                    query_index=case["query_index"],
                    records=case.get("records"),
                    chains=case.get("chains"),
                    first_depth=case.get("first_depth"),
                    second_depth=case.get("second_depth"),
                    distractor_index=distractor_index,
                    distractor_change_score=distractor_change_score,
                    distractor_prefill_seconds=distractor["prefill_seconds"],
                    association_specificity_score=association_specificity_score,
                    association_specificity_accuracy=float(association_specificity_score > 0),
                )
                task_results.append(result)
                results.append(result)
                print(
                    f"{task} {length:,} sample={sample} exact={result['exact_match']:.0f} "
                    f"choice={result['candidate_accuracy']:.0f} "
                    f"score={result['contrastive_retrieval_score']:.3f} "
                    f"specificity={result['association_specificity_score']:.3f}"
                )
        task_summaries[task] = aggregate_results(task_results)
    report = {
        "format": "speck_structured_retrieval_evaluation",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(args.experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": {
            "tasks": list(args.tasks),
            "lengths": list(args.lengths),
            "samples": samples,
            "records": records,
            "chains": chains,
            "template": args.template,
            "answer_set": args.answer_set,
            "response_cue": args.response_cue,
            "two_hop_depths": [list(pair) for pair in TWO_HOP_DEPTHS],
        },
        "positional_regime": positional_regime(
            model,
            configs["train"]["sequence_length"],
            max(args.lengths),
        ),
        "task_summaries": task_summaries,
        "results": results,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "structured-retrieval"
        / configs["train"]["run"]
        / f"{step}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    for task, summary in report["task_summaries"].items():
        curve = summary["curve"]
        short = curve[0]
        long = curve[-1]
        print(
            f"{task}: choice {short['candidate_accuracy']:.3f}->{long['candidate_accuracy']:.3f} "
            f"contrast {short['contrastive_retrieval_score']:.3f}->"
            f"{long['contrastive_retrieval_score']:.3f} "
            f"specificity {short['association_specificity_score']:.3f}->"
            f"{long['association_specificity_score']:.3f}"
        )


if __name__ == "__main__":
    main()
