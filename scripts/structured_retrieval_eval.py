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
from speck.research import load_promotion_protocol, resolve_evaluation_protocol
from speck.tokenizer import get_tokenizer

PRIMARY_TASKS = ("multi_key", "two_hop")
SYMBOLIC_TASK_MODES = {
    "two_hop_route": "route",
    "two_hop_payload": "payload",
    "two_hop_symbolic": "compose",
    "two_hop_chain": "chain",
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
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="freeze all scientific settings from a promotion protocol",
    )
    parser.add_argument(
        "--protocol-length",
        type=int,
        default=None,
        help="run one declared protocol length after its preceding gate has passed",
    )
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
    route_values=None,
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
        route_values=route_values,
    )


def run(args):
    configs = load_experiment(args.experiment, "tokenizer", "train")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    protocol_identity = None
    protocol_length = getattr(args, "protocol_length", None)
    if protocol_path := getattr(args, "protocol", None):
        if protocol_length is None:
            raise ValueError("protocol evaluation requires one explicit --protocol-length")
        loaded_protocol = load_promotion_protocol(protocol_path, tokenizer=tokenizer)
        settings = resolve_evaluation_protocol(
            loaded_protocol, selected_length=protocol_length
        )
        protocol_identity = loaded_protocol["identity"]
    else:
        if protocol_length is not None:
            raise ValueError("--protocol-length requires --protocol")
        samples = positive_integer(args.samples, "samples")
        records = positive_integer(args.records, "records")
        chains = positive_integer(args.chains, "chains")
        settings = {
            "lengths": tuple(args.lengths),
            "samples": samples,
            "seed_offset": 0,
            "kv_cache_dtype": "bfloat16",
            "effective_threshold": 0.85,
            "conditions": tuple(
                {
                    "task": task,
                    "template": args.template,
                    "answer_set": args.answer_set,
                    "records": records,
                    "chains": chains,
                    "response_cue": args.response_cue,
                }
                for task in args.tasks
            ),
            "route_values": None,
        }
    cache_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "int8": torch.int8,
    }
    try:
        cache_dtype = cache_dtypes[settings["kv_cache_dtype"]]
    except KeyError as error:
        raise ValueError("promotion protocol has an unsupported KV cache dtype") from error
    checkpoint_dir = args.checkpoint_dir or Path(
        configs["train"].get("output_dir")
        or Path(base_dir()) / "checkpoints" / configs["train"]["run"]
    )
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    device = torch.device(args.device)
    model, _ = load_checkpoint_model(checkpoint_dir, step, device)
    if max(settings["lengths"]) > model.config.max_position_embeddings:
        raise ValueError("evaluation length exceeds the model's allocated context")
    results = []
    task_summaries = {}
    for condition in settings["conditions"]:
        task = condition["task"]
        condition_name = (
            f"{task}/{condition['template']}/{condition['answer_set']}"
            f"/records_{condition['records']}/chains_{condition['chains']}"
            f"/{condition['response_cue']}_cue"
        )
        task_results = []
        for length in settings["lengths"]:
            for sample in range(settings["samples"]):
                seed = settings["seed_offset"] + sample
                keywords = {
                    "template": condition["template"],
                    "answer_set": condition["answer_set"],
                    "response_cue": condition["response_cue"],
                    "route_values": settings["route_values"],
                }
                case = build_case(
                    task,
                    tokenizer,
                    length,
                    seed,
                    condition["records"],
                    condition["chains"],
                    **keywords,
                )
                counterfactual_case = build_case(
                    task,
                    tokenizer,
                    length,
                    seed,
                    condition["records"],
                    condition["chains"],
                    answer_offset=1,
                    **keywords,
                )
                distractor_index = (case["query_index"] + 1) % (
                    case.get("records") or case["chains"]
                )
                distractor_case = build_case(
                    task,
                    tokenizer,
                    length,
                    seed,
                    condition["records"],
                    condition["chains"],
                    answer_offset=1,
                    mutation_index=distractor_index,
                    **keywords,
                )
                factual = evaluate_case(model, case, device=device, kv_cache_dtype=cache_dtype)
                counterfactual = evaluate_case(
                    model,
                    counterfactual_case,
                    device=device,
                    kv_cache_dtype=cache_dtype,
                )
                result = add_counterfactual_metrics(
                    factual, counterfactual, case, counterfactual_case
                )
                distractor = evaluate_case(
                    model,
                    distractor_case,
                    device=device,
                    kv_cache_dtype=cache_dtype,
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
                    condition=condition_name,
                )
                task_results.append(result)
                results.append(result)
                print(
                    f"{condition_name} {length:,} sample={sample} "
                    f"exact={result['exact_match']:.0f} "
                    f"choice={result['candidate_accuracy']:.0f} "
                    f"score={result['contrastive_retrieval_score']:.3f} "
                    f"specificity={result['association_specificity_score']:.3f}"
                )
        task_summaries[condition_name] = aggregate_results(
            task_results, threshold=settings["effective_threshold"]
        )
    report = {
        "format": "speck_structured_retrieval_evaluation",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(args.experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": {
            "promotion_protocol": protocol_identity,
            "lengths": list(settings["lengths"]),
            "samples_per_condition": settings["samples"],
            "seed_offset": settings["seed_offset"],
            "kv_cache_dtype": settings["kv_cache_dtype"],
            "effective_threshold": settings["effective_threshold"],
            "conditions": list(settings["conditions"]),
            "two_hop_depths": [list(pair) for pair in TWO_HOP_DEPTHS],
        },
        "positional_regime": positional_regime(
            model,
            configs["train"]["sequence_length"],
            max(settings["lengths"]),
        ),
        "task_summaries": task_summaries,
        "results": results,
    }
    output = args.output or Path(base_dir()) / "evaluations" / "structured-retrieval" / configs[
        "train"
    ]["run"] / (
        f"{step}-{protocol_identity['id']}-{settings['lengths'][0]}.json"
        if protocol_identity is not None
        else f"{step}.json"
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
