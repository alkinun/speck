"""Build and score deterministic long-context diagnostics."""

import math
import random
import time
from collections import defaultdict

import torch

ANSWER_SETS = {
    "letters": tuple("ABCDEFGHIJ"),
    "phrases": (
        "silver summit",
        "golden river",
        "north moon",
        "red sun",
        "blue field",
        "green bird",
        "black sky",
        "white rain",
        "orange star",
        "purple house",
    ),
    "phrases_train_v2": (
        "family river",
        "program school",
        "effect light",
        "method group",
        "water house",
        "product model",
        "error report",
        "night service",
        "level change",
        "super array",
        "record table",
        "money market",
        "design sound",
        "reason control",
        "space context",
        "block version",
    ),
    "phrases_validation_v2": (
        "public return",
        "other class",
        "first function",
        "under hand",
        "people work",
        "static value",
        "present course",
        "request data",
        "children play",
        "health life",
        "women person",
        "company account",
        "young family",
        "black sky",
        "front line",
        "known result",
    ),
}
RETRIEVAL_TEMPLATES = ("archive", "registry", "ledger", "manifest", "directory")
ROUTE_VALUES = tuple("KLMNOPQRST")


def parse_lengths(value):
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as error:
        raise ValueError("lengths must be comma-separated integers") from error
    if not lengths or any(length < 32 for length in lengths):
        raise ValueError("context lengths must be at least 32 tokens")
    if tuple(sorted(set(lengths))) != lengths:
        raise ValueError("context lengths must be sorted and unique")
    return lengths


def effective_length(curve, threshold=0.85, metric="exact_match"):
    """Return the longest length retaining the threshold fraction of short performance."""

    if not 0 < threshold <= 1:
        raise ValueError("effective-length threshold must be in (0, 1]")
    if not curve:
        return None
    baseline = curve[0][metric]
    if baseline <= 0:
        return None
    cutoff = threshold * baseline
    retained = [point["length"] for point in curve if point[metric] >= cutoff]
    return max(retained) if retained else None


def binomial_tail_probability(successes, trials, chance):
    """Return P(X >= successes) for a binomial random variable under chance."""

    return sum(
        math.comb(trials, value) * chance**value * (1 - chance) ** (trials - value)
        for value in range(successes, trials + 1)
    )


def _repeat_to_length(pattern, length):
    if not pattern:
        raise ValueError("filler text must produce at least one token")
    repetitions = (length + len(pattern) - 1) // len(pattern)
    return (pattern * repetitions)[:length]


def _answer_values(answer_set):
    try:
        return ANSWER_SETS[answer_set]
    except KeyError as error:
        choices = ", ".join(ANSWER_SETS)
        raise ValueError(f"answer set must be one of: {choices}") from error


def _candidate_tokens(tokenizer, values):
    sequences = [tokenizer.encode(candidate) for candidate in values]
    if any(not sequence for sequence in sequences):
        raise ValueError("retrieval candidates must each encode to at least one token")
    first_tokens = [sequence[0] for sequence in sequences]
    if len(set(first_tokens)) != len(first_tokens):
        raise ValueError("retrieval candidates must have unique first tokens")
    return first_tokens, sequences


def _retrieval_text(
    template,
    task,
    labels,
    answers,
    query_index,
    destinations=None,
    response_cue="native",
):
    if template not in RETRIEVAL_TEMPLATES:
        raise ValueError(f"retrieval template must be one of: {', '.join(RETRIEVAL_TEMPLATES)}")
    if response_cue not in {"native", "answer"}:
        raise ValueError("retrieval response cue must be native or answer")

    def cue(native):
        return "Answer" if response_cue == "answer" else native

    if task == "multi_key":
        if template == "archive":
            return {
                "prefix": "A long archive follows. Remember every exact record and answer the final question.\n",
                "records": [
                    f"The access code for {label} is {answer}."
                    for label, answer in zip(labels, answers)
                ],
                "question": (
                    f"\nQuestion: What is the access code for {labels[query_index]}?\n"
                    f"{cue('Answer')}: "
                ),
                "filler": "The archive contains reports, inventories, correspondence, and notes. ",
            }
        if template == "registry":
            return {
                "prefix": "<REGISTRY>\nThe table below binds identifiers to payloads.\n",
                "records": [
                    f"ID[{label}] :: PAYLOAD[{answer}]" for label, answer in zip(labels, answers)
                ],
                "question": f"\nLOOKUP ID[{labels[query_index]}]\n{cue('PAYLOAD')}: ",
                "filler": "Status nominal. Queue empty. Routine telemetry recorded. ",
            }
        if template == "ledger":
            return {
                "prefix": "A parcel ledger follows. Each named parcel has one seal.\n",
                "records": [
                    f"Parcel {label} bears the seal {answer}."
                    for label, answer in zip(labels, answers)
                ],
                "question": (
                    f"\nWhich seal belongs to parcel {labels[query_index]}?\n{cue('Seal')}: "
                ),
                "filler": "The office logged routine deliveries, receipts, notices, and schedules. ",
            }
        if template == "manifest":
            return {
                "prefix": "A shipping manifest assigns one marker to each labeled shipment.\n",
                "records": [
                    f"Shipment labeled {label} carries marker {answer}."
                    for label, answer in zip(labels, answers)
                ],
                "question": (
                    f"\nReport the marker carried by shipment labeled {labels[query_index]}.\n"
                    f"{cue('Marker')}: "
                ),
                "filler": "Routine freight, invoices, timetables, and maintenance notes were filed. ",
            }
        return {
            "prefix": "A station directory lists the signal assigned to every station.\n",
            "records": [
                f"At station {label}, the assigned signal reads {answer}."
                for label, answer in zip(labels, answers)
            ],
            "question": (
                f"\nState the signal assigned at station {labels[query_index]}.\n{cue('Signal')}: "
            ),
            "filler": "Operators filed weather reports, staffing notes, repairs, and routine notices. ",
        }
    if task != "two_hop" or destinations is None:
        raise ValueError("retrieval text requires a supported task and complete fields")
    if template == "archive":
        return {
            "prefix": (
                "A routed archive follows. Resolve the linked records and answer the final question.\n"
            ),
            "first": [
                f"The route from {label} leads to {destination}."
                for label, destination in zip(labels, destinations)
            ],
            "second": [
                f"The access code inside {destination} is {answer}."
                for destination, answer in zip(destinations, answers)
            ],
            "question": (
                f"\nQuestion: Follow the route from {labels[query_index]}. "
                f"What access code is inside its destination?\n{cue('Answer')}: "
            ),
            "filler": "The archive contains unrelated reports, schedules, and correspondence. ",
        }
    if template == "registry":
        return {
            "prefix": "<ROUTING_TABLE>\nResolve the directed bindings, then return the payload.\n",
            "first": [
                f"EDGE[{label}] -> NODE[{destination}]"
                for label, destination in zip(labels, destinations)
            ],
            "second": [
                f"NODE[{destination}] -> PAYLOAD[{answer}]"
                for destination, answer in zip(destinations, answers)
            ],
            "question": f"\nRESOLVE EDGE[{labels[query_index]}]\n{cue('PAYLOAD')}: ",
            "filler": "Heartbeat stable. No pending route changes. Diagnostic entry complete. ",
        }
    if template == "ledger":
        return {
            "prefix": "A dispatch ledger follows. Each dispatch names a depot holding one seal.\n",
            "first": [
                f"Dispatch {label} names depot {destination}."
                for label, destination in zip(labels, destinations)
            ],
            "second": [
                f"Depot {destination} stores seal {answer}."
                for destination, answer in zip(destinations, answers)
            ],
            "question": (
                f"\nWhich seal is stored at the depot named by dispatch {labels[query_index]}?\n"
                f"{cue('Seal')}: "
            ),
            "filler": "The office logged routine deliveries, receipts, notices, and schedules. ",
        }
    if template == "manifest":
        return {
            "prefix": "A transit manifest links each ticket to a warehouse and cargo marker.\n",
            "first": [
                f"Ticket {label} sends cargo to warehouse {destination}."
                for label, destination in zip(labels, destinations)
            ],
            "second": [
                f"Warehouse {destination} labels its cargo {answer}."
                for destination, answer in zip(destinations, answers)
            ],
            "question": (
                f"\nReport the cargo marker reached from ticket {labels[query_index]}.\n"
                f"{cue('Marker')}: "
            ),
            "filler": "Routine freight, invoices, timetables, and maintenance notes were filed. ",
        }
    return {
        "prefix": "A route directory links each card to a station displaying one signal.\n",
        "first": [
            f"Route card {label} points to station {destination}."
            for label, destination in zip(labels, destinations)
        ],
        "second": [
            f"Station {destination} displays signal {answer}."
            for destination, answer in zip(destinations, answers)
        ],
        "question": (
            f"\nState the signal reached from route card {labels[query_index]}.\n{cue('Signal')}: "
        ),
        "filler": "Operators filed weather reports, staffing notes, repairs, and routine notices. ",
    }


def _exact_prompt(prefix, blocks, question, filler, filler_positions, prompt_length):
    """Interleave ordered blocks with filler at fractional filler positions."""

    fixed = len(prefix) + sum(len(block) for block in blocks) + len(question)
    if fixed > prompt_length:
        raise ValueError("context length is too short for the retrieval template")
    filler_tokens = _repeat_to_length(filler, prompt_length - fixed)
    positions = [round(len(filler_tokens) * depth) for depth in filler_positions]
    prompt = list(prefix)
    previous = 0
    block_positions = []
    for position, block in zip(positions, blocks):
        prompt.extend(filler_tokens[previous:position])
        block_positions.append(len(prompt))
        prompt.extend(block)
        previous = position
    prompt.extend(filler_tokens[previous:])
    prompt.extend(question)
    if len(prompt) != prompt_length:
        raise RuntimeError("retrieval diagnostic did not reach its exact requested length")
    return prompt, block_positions


def build_passkey_case(tokenizer, length, seed, depth, answer_offset=0):
    """Build a literal retrieval case whose prompt and scored answer total an exact length."""

    if not 0 <= depth <= 1:
        raise ValueError("needle depth must be in [0, 1]")
    generator = random.Random(seed)
    label = f"archive-{generator.randrange(100_000, 1_000_000)}"
    values = ANSWER_SETS["letters"]
    answer_index = (generator.randrange(len(values)) + answer_offset) % len(values)
    answer = values[answer_index]
    prefix = tokenizer.encode(
        "A long archive follows. Remember exact records and answer the final question.\n",
        bos=True,
    )
    needle = tokenizer.encode(f"\nThe access code for {label} is {answer}.\n")
    question = tokenizer.encode(f"\nQuestion: What is the access code for {label}?\nAnswer: ")
    answer_tokens = tokenizer.encode(answer)
    candidate_token_ids, candidate_token_sequences = _candidate_tokens(tokenizer, values)
    prompt_length = length - len(answer_tokens)
    fixed = len(prefix) + len(needle) + len(question)
    if fixed > prompt_length:
        raise ValueError(f"context length {length} is too short for the diagnostic template")
    filler_length = prompt_length - fixed
    filler = _repeat_to_length(
        tokenizer.encode(
            "The archive contains ordinary reports, inventories, correspondence, and notes. "
        ),
        filler_length,
    )
    before = round(filler_length * depth)
    prompt = prefix + filler[:before] + needle + filler[before:] + question
    if len(prompt) + len(answer_tokens) != length:
        raise RuntimeError("long-context diagnostic did not reach its exact requested length")
    return {
        "task": "passkey",
        "length": length,
        "depth": depth,
        "seed": seed,
        "prompt_tokens": prompt,
        "prompt_length": len(prompt),
        "answer_tokens": answer_tokens,
        "candidate_token_ids": candidate_token_ids,
        "candidate_token_sequences": candidate_token_sequences,
        "answer": answer,
        "answer_index": answer_index,
        "label": label,
    }


def build_multi_key_case(
    tokenizer,
    length,
    seed,
    depth,
    records=8,
    answer_offset=0,
    mutation_index=None,
    template="archive",
    answer_set="letters",
    response_cue="native",
):
    """Build exact-length associative recall with several simultaneous key/value records."""

    if not 0 <= depth <= 1:
        raise ValueError("record depth must be in [0, 1]")
    if not isinstance(records, int) or isinstance(records, bool) or not 2 <= records <= 10:
        raise ValueError("multi-key records must be an integer in [2, 10]")
    generator = random.Random(seed)
    labels = [f"archive-{generator.randrange(100_000, 1_000_000)}" for _ in range(records)]
    while len(set(labels)) != records:
        labels = [f"archive-{generator.randrange(100_000, 1_000_000)}" for _ in range(records)]
    values = _answer_values(answer_set)
    if records > len(values):
        raise ValueError("multi-key records exceed the selected answer set")
    answers = list(generator.sample(values, records))
    original_answers = list(answers)
    query_index = generator.randrange(records)
    mutation_index = query_index if mutation_index is None else mutation_index
    if not isinstance(mutation_index, int) or isinstance(mutation_index, bool):
        raise ValueError("multi-key mutation index must be an integer")
    if not 0 <= mutation_index < records:
        raise ValueError("multi-key mutation index is outside the records")
    answers[mutation_index] = values[
        (values.index(answers[mutation_index]) + answer_offset) % len(values)
    ]
    text = _retrieval_text(
        template,
        "multi_key",
        labels,
        answers,
        query_index,
        response_cue=response_cue,
    )
    prefix = tokenizer.encode(text["prefix"], bos=True)
    record_lines = list(zip(labels, answers))
    generator.shuffle(record_lines)
    shuffled_text = _retrieval_text(
        template,
        "multi_key",
        [label for label, _ in record_lines],
        [answer for _, answer in record_lines],
        0,
        response_cue=response_cue,
    )
    record_block = tokenizer.encode("\n".join(shuffled_text["records"]) + "\n")
    question = tokenizer.encode(text["question"])
    answer = answers[query_index]
    answer_tokens = tokenizer.encode(answer)
    candidate_token_ids, candidate_token_sequences = _candidate_tokens(tokenizer, values)
    prompt, positions = _exact_prompt(
        prefix,
        (record_block,),
        question,
        tokenizer.encode(text["filler"]),
        (depth,),
        length - len(answer_tokens),
    )
    return {
        "task": "multi_key",
        "length": length,
        "depth": depth,
        "seed": seed,
        "prompt_tokens": prompt,
        "prompt_length": len(prompt),
        "answer_tokens": answer_tokens,
        "candidate_token_ids": candidate_token_ids,
        "candidate_token_sequences": candidate_token_sequences,
        "answer": answer,
        "answer_index": values.index(answer),
        "label": labels[query_index],
        "query_index": query_index,
        "mutation_index": mutation_index,
        "mutation_from_index": values.index(original_answers[mutation_index]),
        "mutation_to_index": values.index(answers[mutation_index]),
        "records": records,
        "fact_positions": positions,
        "template": template,
        "answer_set": answer_set,
        "response_cue": response_cue,
    }


def build_two_hop_case(
    tokenizer,
    length,
    seed,
    first_depth,
    second_depth,
    chains=6,
    answer_offset=0,
    mutation_index=None,
    template="archive",
    answer_set="letters",
    response_cue="native",
):
    """Build exact-length two-hop lookup among several independent chains."""

    if not 0 <= first_depth < second_depth <= 1:
        raise ValueError("two-hop depths must satisfy 0 <= first < second <= 1")
    if not isinstance(chains, int) or isinstance(chains, bool) or not 2 <= chains <= 10:
        raise ValueError("two-hop chains must be an integer in [2, 10]")
    generator = random.Random(seed)
    starts = [f"index-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
    destinations = [f"box-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
    while len(set(starts)) != chains or len(set(destinations)) != chains:
        starts = [f"index-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
        destinations = [f"box-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
    values = _answer_values(answer_set)
    if chains > len(values):
        raise ValueError("two-hop chains exceed the selected answer set")
    answers = list(generator.sample(values, chains))
    original_answers = list(answers)
    query_index = generator.randrange(chains)
    mutation_index = query_index if mutation_index is None else mutation_index
    if not isinstance(mutation_index, int) or isinstance(mutation_index, bool):
        raise ValueError("two-hop mutation index must be an integer")
    if not 0 <= mutation_index < chains:
        raise ValueError("two-hop mutation index is outside the chains")
    answers[mutation_index] = values[
        (values.index(answers[mutation_index]) + answer_offset) % len(values)
    ]
    first_lines = list(zip(starts, destinations))
    second_lines = list(zip(destinations, answers))
    generator.shuffle(first_lines)
    generator.shuffle(second_lines)
    text = _retrieval_text(
        template,
        "two_hop",
        starts,
        answers,
        query_index,
        destinations=destinations,
        response_cue=response_cue,
    )
    first_text = _retrieval_text(
        template,
        "two_hop",
        [start for start, _ in first_lines],
        answers,
        query_index,
        destinations=[destination for _, destination in first_lines],
        response_cue=response_cue,
    )
    answer_by_destination = dict(zip(destinations, answers))
    second_text = _retrieval_text(
        template,
        "two_hop",
        starts,
        [answer_by_destination[destination] for destination, _ in second_lines],
        query_index,
        destinations=[destination for destination, _ in second_lines],
        response_cue=response_cue,
    )
    first_block = tokenizer.encode("\n".join(first_text["first"]) + "\n")
    second_block = tokenizer.encode("\n".join(second_text["second"]) + "\n")
    prefix = tokenizer.encode(text["prefix"], bos=True)
    question = tokenizer.encode(text["question"])
    answer = answers[query_index]
    answer_tokens = tokenizer.encode(answer)
    candidate_token_ids, candidate_token_sequences = _candidate_tokens(tokenizer, values)
    prompt, positions = _exact_prompt(
        prefix,
        (first_block, second_block),
        question,
        tokenizer.encode(text["filler"]),
        (first_depth, second_depth),
        length - len(answer_tokens),
    )
    return {
        "task": "two_hop",
        "length": length,
        "depth": second_depth,
        "first_depth": first_depth,
        "second_depth": second_depth,
        "seed": seed,
        "prompt_tokens": prompt,
        "prompt_length": len(prompt),
        "answer_tokens": answer_tokens,
        "candidate_token_ids": candidate_token_ids,
        "candidate_token_sequences": candidate_token_sequences,
        "answer": answer,
        "answer_index": values.index(answer),
        "label": starts[query_index],
        "query_index": query_index,
        "mutation_index": mutation_index,
        "mutation_from_index": values.index(original_answers[mutation_index]),
        "mutation_to_index": values.index(answers[mutation_index]),
        "chains": chains,
        "fact_positions": positions,
        "template": template,
        "answer_set": answer_set,
        "response_cue": response_cue,
    }


def build_symbolic_two_hop_case(
    tokenizer,
    length,
    seed,
    first_depth,
    second_depth,
    chains=6,
    answer_offset=0,
    mutation_index=None,
    mode="compose",
    template="archive",
    answer_set="letters",
    response_cue="native",
    route_values=None,
):
    """Build route, payload, or composed lookup over a declared intermediate vocabulary."""

    if mode not in {"route", "payload", "compose", "chain"}:
        raise ValueError("symbolic two-hop mode must be route, payload, compose, or chain")
    if not 0 <= first_depth < second_depth <= 1:
        raise ValueError("symbolic two-hop depths must satisfy 0 <= first < second <= 1")
    if not isinstance(chains, int) or isinstance(chains, bool) or not 2 <= chains <= 9:
        raise ValueError("symbolic two-hop chains must be an integer in [2, 9]")
    values = _answer_values(answer_set)
    if chains > len(values):
        raise ValueError("symbolic two-hop chains exceed the selected answer set")
    route_values = tuple(ROUTE_VALUES if route_values is None else route_values)
    if len(route_values) <= chains or len(set(route_values)) != len(route_values):
        raise ValueError("symbolic route values must be unique and outnumber the active chains")
    if any(not isinstance(value, str) or not value for value in route_values):
        raise ValueError("symbolic route values must be non-empty strings")
    generator = random.Random(seed)
    starts = [f"index-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
    while len(set(starts)) != chains:
        starts = [f"index-{generator.randrange(100_000, 1_000_000)}" for _ in range(chains)]
    destinations = list(generator.sample(route_values, chains))
    answers = list(generator.sample(values, chains))
    query_index = generator.randrange(chains)
    mutation_index = query_index if mutation_index is None else mutation_index
    if not isinstance(mutation_index, int) or isinstance(mutation_index, bool):
        raise ValueError("symbolic two-hop mutation index must be an integer")
    if not 0 <= mutation_index < chains:
        raise ValueError("symbolic two-hop mutation index is outside the chains")
    if mode in {"route", "chain"}:
        candidate_values = route_values
        original_answer = destinations[mutation_index]
        if answer_offset:
            unused = [value for value in route_values if value not in destinations]
            destinations[mutation_index] = unused[(answer_offset - 1) % len(unused)]
        mutated_answer = destinations[mutation_index]
    else:
        candidate_values = values
        original_answer = answers[mutation_index]
        answers[mutation_index] = values[
            (values.index(answers[mutation_index]) + answer_offset) % len(values)
        ]
        mutated_answer = answers[mutation_index]

    first_lines = list(zip(starts, destinations))
    second_lines = list(zip(destinations, answers))
    generator.shuffle(first_lines)
    generator.shuffle(second_lines)
    text = _retrieval_text(
        template,
        "two_hop",
        starts,
        answers,
        query_index,
        destinations=destinations,
        response_cue=response_cue,
    )
    first_text = _retrieval_text(
        template,
        "two_hop",
        [start for start, _ in first_lines],
        answers,
        query_index,
        destinations=[destination for _, destination in first_lines],
        response_cue=response_cue,
    )
    answer_by_destination = dict(zip(destinations, answers))
    second_text = _retrieval_text(
        template,
        "two_hop",
        starts,
        [answer_by_destination[destination] for destination, _ in second_lines],
        query_index,
        destinations=[destination for destination, _ in second_lines],
        response_cue=response_cue,
    )
    first_block = tokenizer.encode("\n".join(first_text["first"]) + "\n")
    second_block = tokenizer.encode("\n".join(second_text["second"]) + "\n")
    prefix = tokenizer.encode(text["prefix"], bos=True)
    if mode == "route":
        native_cue = "Answer" if response_cue == "answer" else "Destination"
        question_text = (
            f"\nQuestion: Follow the route from {starts[query_index]}. "
            f"Which destination does it reach?\n{native_cue}: "
        )
        answer = destinations[query_index]
    elif mode == "chain":
        question_text = (
            f"\nQuestion: Follow the route from {starts[query_index]}. "
            "Respond with its destination followed by the access code there.\nAnswer: "
        )
        answer = f"{destinations[query_index]} {answers[query_index]}"
    elif mode == "payload":
        native_cue = "Answer" if response_cue == "answer" else "Code"
        question_text = (
            f"\nQuestion: What access code is stored at destination "
            f"{destinations[query_index]}?\n{native_cue}: "
        )
        answer = answers[query_index]
    else:
        question_text = text["question"]
        answer = answers[query_index]
    question = tokenizer.encode(question_text)
    answer_tokens = tokenizer.encode(answer)
    candidate_token_ids, candidate_token_sequences = _candidate_tokens(tokenizer, candidate_values)
    prompt, positions = _exact_prompt(
        prefix,
        (first_block, second_block),
        question,
        tokenizer.encode(text["filler"]),
        (first_depth, second_depth),
        length - len(answer_tokens),
    )
    task = {
        "route": "two_hop_route",
        "payload": "two_hop_payload",
        "compose": "two_hop_symbolic",
        "chain": "two_hop_chain",
    }[mode]
    return {
        "task": task,
        "length": length,
        "depth": second_depth,
        "first_depth": first_depth,
        "second_depth": second_depth,
        "seed": seed,
        "prompt_tokens": prompt,
        "prompt_length": len(prompt),
        "answer_tokens": answer_tokens,
        "candidate_token_ids": candidate_token_ids,
        "candidate_token_sequences": candidate_token_sequences,
        "answer": answer,
        "answer_index": candidate_values.index(
            destinations[query_index] if mode == "chain" else answer
        ),
        "label": starts[query_index],
        "query_index": query_index,
        "mutation_index": mutation_index,
        "mutation_from_index": candidate_values.index(original_answer),
        "mutation_to_index": candidate_values.index(mutated_answer),
        "chains": chains,
        "fact_positions": positions,
        "template": template,
        "answer_set": answer_set,
        "response_cue": response_cue,
        "symbolic_mode": mode,
        "destination": destinations[query_index],
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_case(model, case, device=None, state_dtype=None, kv_cache_dtype=None):
    """Score full answer emission and first-token candidate choice without full-sequence logits."""

    parameter = next(model.parameters())
    device = torch.device(device or parameter.device)
    prompt = torch.tensor([case["prompt_tokens"]], device=device)
    answers = case["answer_tokens"]
    state = model.state(
        length=case["length"],
        device=device,
        dtype=state_dtype,
        kv_cache_dtype=kv_cache_dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    logits = model(prompt, state=state, last_token_only=True)[:, -1]
    _synchronize(device)
    prefill_seconds = time.perf_counter() - started
    candidate_ids = torch.tensor(case["candidate_token_ids"], device=device)
    candidate_logits = logits[0, candidate_ids].float()
    correct_candidate = case["candidate_token_ids"].index(answers[0])
    correct_logit = candidate_logits[correct_candidate]
    other_logits = torch.cat(
        (candidate_logits[:correct_candidate], candidate_logits[correct_candidate + 1 :])
    )
    candidate_rank = int((candidate_logits > correct_logit).sum().item()) + 1
    candidate_prediction = int(candidate_logits.argmax().item())
    predictions = []
    log_probabilities = []
    decode_started = time.perf_counter()
    for token_id in answers:
        log_probabilities.append(torch.log_softmax(logits.float(), dim=-1)[0, token_id].item())
        predictions.append(logits.argmax(dim=-1).item())
        token = torch.tensor([[token_id]], device=device)
        logits = model(token, state=state, last_token_only=True)[:, -1]
    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    matched = sum(expected == actual for expected, actual in zip(answers, predictions))
    memory = state.memory_report()
    return {
        "task": case["task"],
        "length": case["length"],
        "depth": case["depth"],
        "seed": case["seed"],
        "prompt_tokens": prompt.size(1),
        "answer_tokens": len(answers),
        "exact_match": float(matched == len(answers)),
        "token_accuracy": matched / len(answers),
        "candidate_accuracy": float(candidate_prediction == correct_candidate),
        "candidate_count": len(case["candidate_token_ids"]),
        "candidate_probability": candidate_logits.softmax(dim=0)[correct_candidate].item(),
        "candidate_log_probabilities": candidate_logits.log_softmax(dim=0).tolist(),
        "candidate_rank": candidate_rank,
        "candidate_margin": (correct_logit - other_logits.max()).item(),
        "mean_log_probability": sum(log_probabilities) / len(log_probabilities),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt.size(1) / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": len(answers) / decode_seconds,
        "state_memory": memory,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def add_counterfactual_metrics(factual, counterfactual, factual_case, counterfactual_case):
    """Attach paired prompt-sensitivity metrics to an evaluated factual case."""

    factual_index = factual_case["answer_index"]
    counterfactual_index = counterfactual_case["answer_index"]
    factual_scores = factual["candidate_log_probabilities"]
    counterfactual_scores = counterfactual["candidate_log_probabilities"]
    factual_preference = factual_scores[factual_index] - factual_scores[counterfactual_index]
    counterfactual_preference = (
        counterfactual_scores[counterfactual_index] - counterfactual_scores[factual_index]
    )
    factual.update(
        counterfactual_answer=counterfactual_case["answer"],
        counterfactual_prefill_seconds=counterfactual["prefill_seconds"],
        contrastive_retrieval_score=(factual_preference + counterfactual_preference) / 2,
        contrastive_direction_accuracy=float(factual_preference + counterfactual_preference > 0),
        contrastive_pair_accuracy=float(factual_preference > 0 and counterfactual_preference > 0),
    )
    return factual


def candidate_shift_score(reference, changed, from_index, to_index):
    """Measure symmetric candidate-logit movement caused by one prompt mutation."""

    reference_scores = reference["candidate_log_probabilities"]
    changed_scores = changed["candidate_log_probabilities"]
    reference_preference = reference_scores[from_index] - reference_scores[to_index]
    changed_preference = changed_scores[to_index] - changed_scores[from_index]
    return (reference_preference + changed_preference) / 2


def aggregate_results(results, threshold=0.85):
    grouped = defaultdict(list)
    for result in results:
        grouped[result["length"]].append(result)
    curve = []
    metrics = (
        "exact_match",
        "token_accuracy",
        "candidate_accuracy",
        "candidate_probability",
        "candidate_rank",
        "candidate_margin",
        "mean_log_probability",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    )
    has_contrastive = results and all("contrastive_retrieval_score" in result for result in results)
    has_specificity = results and all(
        "association_specificity_score" in result for result in results
    )
    if has_contrastive:
        metrics += (
            "contrastive_retrieval_score",
            "contrastive_direction_accuracy",
            "contrastive_pair_accuracy",
            "counterfactual_prefill_seconds",
        )
    if has_specificity:
        metrics += (
            "distractor_change_score",
            "association_specificity_score",
            "association_specificity_accuracy",
            "distractor_prefill_seconds",
        )
    for length, values in sorted(grouped.items()):
        point = {"length": length, "samples": len(values)}
        for metric in metrics:
            point[metric] = sum(value[metric] for value in values) / len(values)
        point["state_bytes"] = max(value["state_memory"]["total_bytes"] for value in values)
        point["state_by_kind"] = values[0]["state_memory"]["by_kind"]
        candidate_counts = {value["candidate_count"] for value in values}
        if len(candidate_counts) != 1:
            raise ValueError("candidate count changed within a context length")
        point["candidate_successes"] = sum(value["candidate_accuracy"] for value in values)
        point["candidate_trials"] = len(values)
        point["candidate_chance_accuracy"] = 1 / candidate_counts.pop()
        peak_values = [value["peak_allocated_bytes"] for value in values]
        point["peak_allocated_bytes"] = (
            max(value for value in peak_values if value is not None)
            if any(value is not None for value in peak_values)
            else None
        )
        curve.append(point)
    candidate_p_value = None
    candidate_effective_length = None
    contrastive_p_value = None
    contrastive_effective_length = None
    specificity_p_value = None
    specificity_effective_length = None
    if curve:
        baseline = curve[0]
        candidate_p_value = binomial_tail_probability(
            int(baseline["candidate_successes"]),
            baseline["candidate_trials"],
            baseline["candidate_chance_accuracy"],
        )
        if candidate_p_value < 0.05:
            candidate_effective_length = effective_length(
                curve, threshold, metric="candidate_accuracy"
            )
        if "contrastive_direction_accuracy" in baseline:
            contrastive_successes = round(
                baseline["contrastive_direction_accuracy"] * baseline["samples"]
            )
            contrastive_p_value = binomial_tail_probability(
                contrastive_successes,
                baseline["samples"],
                0.5,
            )
            if contrastive_p_value < 0.05:
                contrastive_effective_length = effective_length(
                    curve,
                    threshold,
                    metric="contrastive_direction_accuracy",
                )
        if "association_specificity_accuracy" in baseline:
            specificity_successes = round(
                baseline["association_specificity_accuracy"] * baseline["samples"]
            )
            specificity_p_value = binomial_tail_probability(
                specificity_successes,
                baseline["samples"],
                0.5,
            )
            if specificity_p_value < 0.05:
                specificity_effective_length = effective_length(
                    curve,
                    threshold,
                    metric="association_specificity_accuracy",
                )
    return {
        "curve": curve,
        "effective_length": effective_length(curve, threshold),
        "effective_length_by_candidate_accuracy": candidate_effective_length,
        "effective_length_threshold": threshold,
        "short_context_baseline": curve[0]["exact_match"] if curve else None,
        "short_context_candidate_baseline": curve[0]["candidate_accuracy"] if curve else None,
        "short_context_candidate_p_value": candidate_p_value,
        "effective_length_by_contrastive_retrieval": contrastive_effective_length,
        "short_context_contrastive_p_value": contrastive_p_value,
        "effective_length_by_association_specificity": specificity_effective_length,
        "short_context_association_specificity_p_value": specificity_p_value,
    }


def validate_eval_settings(settings):
    required = {"lengths", "depths", "samples_per_depth", "effective_threshold"}
    allowed = required | {"kv_cache_dtype"}
    if not required <= set(settings) or set(settings) - allowed:
        raise ValueError("long-context settings have missing or unknown fields")
    lengths = settings["lengths"]
    if not isinstance(lengths, list):
        raise ValueError("long-context lengths must be a list")
    parse_lengths(",".join(map(str, lengths)))
    depths = settings["depths"]
    if (
        not isinstance(depths, list)
        or not depths
        or any(
            isinstance(depth, bool) or not isinstance(depth, (int, float)) or not 0 <= depth <= 1
            for depth in depths
        )
    ):
        raise ValueError("long-context depths must be numbers in [0, 1]")
    samples = settings["samples_per_depth"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples_per_depth must be a positive integer")
    threshold = settings["effective_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("effective_threshold must be numeric")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("effective_threshold must be in (0, 1]")
    kv_cache_dtype = settings.get("kv_cache_dtype", "bfloat16")
    if kv_cache_dtype not in {"bfloat16", "float16", "float32", "int8"}:
        raise ValueError("unsupported long-context KV cache dtype")
    return {
        "lengths": tuple(lengths),
        "depths": tuple(float(depth) for depth in depths),
        "samples_per_depth": samples,
        "effective_threshold": float(threshold),
        "kv_cache_dtype": kv_cache_dtype,
    }
