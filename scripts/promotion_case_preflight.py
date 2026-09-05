"""Preflight every frozen promotion condition without loading a model checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.structured_retrieval_eval import build_case
from speck.architecture import canonical_json
from speck.config import load_experiment
from speck.research import (
    load_promotion_protocol,
    resolve_adaptation_protocol,
    resolve_evaluation_protocol,
)
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--tokenizer-experiment", type=Path, required=True)
    return parser.parse_args(argv)


def _hash_case(digest, kind, case):
    payload = {
        "kind": kind,
        "task": case["task"],
        "length": case["length"],
        "seed": case["seed"],
        "prompt_tokens": case["prompt_tokens"],
        "answer_tokens": case["answer_tokens"],
        "candidate_token_ids": case["candidate_token_ids"],
        "answer_index": case["answer_index"],
        "query_index": case["query_index"],
        "mutation_index": case["mutation_index"],
        "mutation_from_index": case["mutation_from_index"],
        "mutation_to_index": case["mutation_to_index"],
        "template": case["template"],
        "answer_set": case["answer_set"],
        "response_cue": case["response_cue"],
    }
    digest.update(canonical_json(payload).encode())
    digest.update(b"\n")


def preflight_protocol(protocol_path, tokenizer, samples_override=None):
    """Build the shortest exact-length cell and hash every declared validation case."""

    loaded = load_promotion_protocol(protocol_path, tokenizer=tokenizer)
    evaluation = resolve_evaluation_protocol(loaded)
    declared_samples = evaluation["samples"]
    samples = declared_samples if samples_override is None else samples_override
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 1
        or samples > declared_samples
    ):
        raise ValueError("preflight sample override must be within the declared sample count")
    shortest_length = min(evaluation["lengths"])
    conditions = []
    total_cases = 0
    for condition in evaluation["conditions"]:
        digest = hashlib.sha256()
        candidate_counts = set()
        answer_lengths = set()
        query_indices = set()
        for sample in range(samples):
            seed = evaluation["seed_offset"] + sample
            keywords = {
                "template": condition["template"],
                "answer_set": condition["answer_set"],
                "response_cue": condition["response_cue"],
                "route_values": evaluation["route_values"],
            }
            factual = build_case(
                condition["task"],
                tokenizer,
                shortest_length,
                seed,
                condition["records"],
                condition["chains"],
                **keywords,
            )
            counterfactual = build_case(
                condition["task"],
                tokenizer,
                shortest_length,
                seed,
                condition["records"],
                condition["chains"],
                answer_offset=1,
                **keywords,
            )
            distractor_index = (factual["query_index"] + 1) % (
                factual.get("records") or factual["chains"]
            )
            distractor = build_case(
                condition["task"],
                tokenizer,
                shortest_length,
                seed,
                condition["records"],
                condition["chains"],
                answer_offset=1,
                mutation_index=distractor_index,
                **keywords,
            )
            for case in (factual, counterfactual, distractor):
                if len(case["prompt_tokens"]) + len(case["answer_tokens"]) != shortest_length:
                    raise RuntimeError("promotion case does not match its exact declared length")
                if case["candidate_token_ids"] != factual["candidate_token_ids"]:
                    raise RuntimeError(
                        "promotion case candidate vocabulary changed across controls"
                    )
            if (
                factual["label"] != counterfactual["label"]
                or factual["query_index"] != counterfactual["query_index"]
            ):
                raise RuntimeError("target counterfactual changed non-target structure")
            if factual["answer_index"] == counterfactual["answer_index"]:
                raise RuntimeError("target counterfactual did not change the selected answer")
            if factual["answer_index"] != distractor["answer_index"]:
                raise RuntimeError("distractor mutation changed the queried answer")
            for kind, case in (
                ("factual", factual),
                ("counterfactual", counterfactual),
                ("distractor", distractor),
            ):
                _hash_case(digest, kind, case)
            candidate_counts.add(len(factual["candidate_token_ids"]))
            answer_lengths.add(len(factual["answer_tokens"]))
            query_indices.add(factual["query_index"])
            total_cases += 3
        if len(candidate_counts) != 1 or len(answer_lengths) != 1:
            raise RuntimeError("promotion condition changes candidate or answer geometry")
        conditions.append(
            {
                **condition,
                "length": shortest_length,
                "samples": samples,
                "generated_cases": 3 * samples,
                "candidate_count": candidate_counts.pop(),
                "answer_token_length": answer_lengths.pop(),
                "observed_query_indices": sorted(query_indices),
                "case_stream_sha256": digest.hexdigest(),
            }
        )
    adaptation = {
        str(seed): {
            "train_seed_offset": settings["train_seed_offset"],
            "validation_seed_offset": settings["validation_seed_offset"],
            "validation_samples": settings["validation_samples"],
            "train_record_counts": list(settings["train_record_counts"]),
            "validation_record_counts": list(settings["validation_record_counts"]),
        }
        for seed in loaded["protocol"]["comparison"]["base_model_seeds"]
        for settings in (resolve_adaptation_protocol(loaded, seed),)
    }
    return {
        "format": "speck_promotion_case_preflight",
        "format_version": 1,
        "status": "complete" if samples == declared_samples else "test_subset",
        "protocol": loaded["identity"],
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "declared_samples_per_condition": declared_samples,
        "evaluated_samples_per_condition": samples,
        "shortest_declared_length": shortest_length,
        "conditions": conditions,
        "adaptation_seed_plan": adaptation,
        "total_generated_cases": total_cases,
    }


def main(argv=None):
    args = arguments(argv)
    tokenizer_config = load_experiment(args.tokenizer_experiment, "tokenizer")["tokenizer"]
    tokenizer = get_tokenizer(**tokenizer_config)
    print(json.dumps(preflight_protocol(args.protocol, tokenizer), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
