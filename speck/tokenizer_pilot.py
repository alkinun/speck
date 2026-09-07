"""Analyze the frozen matched-tokenizer LM pilot without opening its D5 audit."""

import math
import random
from statistics import median

FORMAT_VERSION = 1
RUN_FORMAT = "speck_tokenizer_pilot_run"
RESULT_FORMAT = "speck_tokenizer_pilot_analysis"
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")
VIEWS = ("fixed_document", "fixed_flop")


def validate_pilot_plan(plan):
    """Validate the complete pre-results pilot and statistical contract."""

    if (
        not isinstance(plan, dict)
        or plan.get("format") != "speck_tokenizer_pilot_plan"
        or plan.get("format_version") != FORMAT_VERSION
        or plan.get("status") != "analysis_fixture_ready_real_runs_blocked"
        or plan.get("categories") != list(CATEGORIES)
        or plan.get("primary_baseline") != "mistral-32k"
        or plan.get("screen", {}).get("seed") != 42
        or plan.get("screen", {}).get("runs")
        != ["mistral-32k", "compression_endpoint", "compact_endpoint"]
        or plan.get("confirmation", {}).get("seeds") != [42, 43, 44]
        or plan.get("confirmation", {}).get("total_runs") != 7
        or plan.get("stopping", {}).get("fixed_document_mistral_tokens") != 1_200_000_000
        or plan.get("statistics", {}).get("bootstrap_seed") != 314159
        or plan.get("statistics", {}).get("bootstrap_replicates") != 10_000
        or plan.get("statistics", {}).get("upper_percentile") != 0.95
        or plan.get("eligibility", {}).get("required_views") != list(VIEWS)
        or plan.get("eligibility", {}).get("aggregate_upper_95_bpb") != 0.01
        or plan.get("eligibility", {}).get("category_upper_95_bpb") != 0.02
        or plan.get("sealed_audit", {}).get("identity") != "D5_tokenizer"
        or plan.get("sealed_audit", {}).get("required_finalists") != 2
        or plan.get("sealed_audit", {}).get("additional_search_after_failure") is not False
        or plan.get("gpu_hour_ceiling") != 30
        or plan.get("final_selection_authority") is not False
    ):
        raise ValueError("tokenizer pilot plan differs from the frozen contract")
    return plan


def _bpb(document):
    size = document.get("utf8_bytes")
    nll = document.get("nll_nats")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or isinstance(nll, bool)
        or not isinstance(nll, (int, float))
        or not math.isfinite(nll)
        or nll < 0
    ):
        raise ValueError("pilot document has invalid bytes or NLL")
    return float(nll) / (math.log(2) * size)


def _validate_run(run, plan):
    if (
        run.get("format") != RUN_FORMAT
        or run.get("format_version") != FORMAT_VERSION
        or run.get("status") != "complete"
        or not isinstance(run.get("tokenizer_id"), str)
        or run.get("seed") not in plan["confirmation"]["seeds"]
        or not isinstance(run.get("tokenizer_model_sha256"), str)
        or not isinstance(run.get("model_manifest_sha256"), str)
        or not isinstance(run.get("backbone_manifest_sha256"), str)
        or not isinstance(run.get("document_stream_sha256"), str)
        or not isinstance(run.get("vocab_size"), int)
        or run["vocab_size"] < 1
        or isinstance(run.get("total_parameters"), bool)
        or not isinstance(run.get("total_parameters"), int)
        or run["total_parameters"] < 1
    ):
        raise ValueError("invalid tokenizer pilot run identity")
    for view in VIEWS:
        value = run.get(view)
        if not isinstance(value, dict) or set(value) != {
            "mistral_reference_tokens",
            "tokenizer_tokens",
            "analytic_flops",
            "target_analytic_flops",
            "active_seconds",
            "peak_memory_bytes",
            "throughput_tokens_per_second",
            "categories",
        }:
            raise ValueError(f"pilot run has invalid {view} view")
        for key in (
            "mistral_reference_tokens",
            "tokenizer_tokens",
            "analytic_flops",
            "target_analytic_flops",
            "active_seconds",
            "peak_memory_bytes",
            "throughput_tokens_per_second",
        ):
            number = value[key]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number < 0
            ):
                raise ValueError(f"pilot run {view} {key} is invalid")
        if value["mistral_reference_tokens"] != plan["stopping"]["fixed_document_mistral_tokens"]:
            raise ValueError("pilot run has wrong fixed-document reference horizon")
        if set(value["categories"]) != set(CATEGORIES):
            raise ValueError("pilot view must report every category")
        for category in CATEGORIES:
            documents = value["categories"][category]
            if not isinstance(documents, list) or not documents:
                raise ValueError("pilot category documents must be non-empty")
            ids = []
            for document in documents:
                if not isinstance(document, dict) or set(document) != {
                    "document_id",
                    "utf8_bytes",
                    "nll_nats",
                }:
                    raise ValueError("pilot document schema is invalid")
                if not isinstance(document["document_id"], str) or not document["document_id"]:
                    raise ValueError("pilot document ID must be non-empty")
                ids.append(document["document_id"])
                _bpb(document)
            if len(ids) != len(set(ids)):
                raise ValueError("pilot category document IDs must be unique")
    curve = run.get("learning_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("pilot run learning curve must be non-empty")
    previous_flops = -1
    for point in curve:
        if not isinstance(point, dict) or set(point) != {
            "analytic_flops",
            "active_seconds",
            "macro_bpb",
        }:
            raise ValueError("pilot learning curve point is invalid")
        if (
            any(
                isinstance(point[key], bool)
                or not isinstance(point[key], (int, float))
                or not math.isfinite(point[key])
                or point[key] < 0
                for key in point
            )
            or point["analytic_flops"] <= previous_flops
        ):
            raise ValueError("pilot learning curve must be finite and increasing")
        previous_flops = point["analytic_flops"]
    if not math.isclose(curve[-1]["macro_bpb"], _macro(run, "fixed_document"), abs_tol=1e-12):
        raise ValueError("pilot learning curve does not end at fixed-document macro BPB")
    return run


def _category_means(run, view):
    return {
        category: sum(_bpb(document) for document in run[view]["categories"][category])
        / len(run[view]["categories"][category])
        for category in CATEGORIES
    }


def _macro(run, view):
    means = _category_means(run, view)
    return sum(means.values()) / len(means)


def _paired_document_deltas(candidate_runs, baseline_runs, view):
    category_deltas = {}
    seed_means = {}
    for category in CATEGORIES:
        per_seed = {}
        expected_ids = None
        expected_bytes = None
        for candidate, baseline in zip(candidate_runs, baseline_runs):
            candidate_docs = {
                document["document_id"]: document
                for document in candidate[view]["categories"][category]
            }
            baseline_docs = {
                document["document_id"]: document
                for document in baseline[view]["categories"][category]
            }
            ids = tuple(sorted(candidate_docs))
            bytes_map = {key: candidate_docs[key]["utf8_bytes"] for key in ids}
            if (
                set(candidate_docs) != set(baseline_docs)
                or any(
                    candidate_docs[key]["utf8_bytes"] != baseline_docs[key]["utf8_bytes"]
                    for key in ids
                )
                or (expected_ids is not None and ids != expected_ids)
                or (expected_bytes is not None and bytes_map != expected_bytes)
            ):
                raise ValueError("pilot candidate and baseline documents are not exactly paired")
            expected_ids = ids
            expected_bytes = bytes_map
            values = {key: _bpb(candidate_docs[key]) - _bpb(baseline_docs[key]) for key in ids}
            per_seed[candidate["seed"]] = values
        averaged = {
            key: sum(per_seed[seed][key] for seed in sorted(per_seed)) / len(per_seed)
            for key in expected_ids
        }
        category_deltas[category] = averaged
        seed_means[category] = {
            str(seed): sum(values.values()) / len(values) for seed, values in per_seed.items()
        }
    return category_deltas, seed_means


def _percentile(values, fraction):
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _bootstrap(category_deltas, plan, view):
    rng = random.Random(f"{plan['statistics']['bootstrap_seed']}:{view}")
    category_samples = {category: [] for category in CATEGORIES}
    macro_samples = []
    for _ in range(plan["statistics"]["bootstrap_replicates"]):
        means = []
        for category in CATEGORIES:
            values = list(category_deltas[category].values())
            mean = sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            category_samples[category].append(mean)
            means.append(mean)
        macro_samples.append(sum(means) / len(means))
    fraction = plan["statistics"]["upper_percentile"]
    observed_categories = {
        category: sum(values.values()) / len(values) for category, values in category_deltas.items()
    }
    return {
        "view": view,
        "macro_delta_bpb": sum(observed_categories.values()) / len(observed_categories),
        "macro_upper_95_bpb": _percentile(macro_samples, fraction),
        "categories": {
            category: {
                "delta_bpb": observed_categories[category],
                "upper_95_bpb": _percentile(category_samples[category], fraction),
            }
            for category in CATEGORIES
        },
    }


def _flops_to_quality(run, target):
    for point in run["learning_curve"]:
        if point["macro_bpb"] <= target:
            return point["analytic_flops"]
    return None


def _seconds_to_quality(run, target):
    for point in run["learning_curve"]:
        if point["macro_bpb"] <= target:
            return point["active_seconds"]
    return None


def _bpb_at_wallclock(run, seconds):
    eligible = [point for point in run["learning_curve"] if point["active_seconds"] <= seconds]
    return (eligible[-1] if eligible else run["learning_curve"][0])["macro_bpb"]


def analyze_tokenizer_pilot(plan, nominations, runs, *, fixture=False):
    """Screen, confirm, gate, rank, and prepare—but do not open—the D5 audit."""

    plan = validate_pilot_plan(plan)
    expected_nomination_status = (
        "fixture_endpoints_reported_no_advancement_authority"
        if fixture
        else "real_static_endpoints_nominated_for_lm_pilot_no_selection_authority"
    )
    if (
        nominations.get("format") != "speck_tokenizer_static_nomination"
        or nominations.get("status") != expected_nomination_status
        or nominations.get("selection_authority") is not False
        or len(nominations.get("nominations", [])) != 2
    ):
        raise ValueError("pilot requires the exact frozen static nomination output")
    custom_ids = [value["id"] for value in nominations["nominations"]]
    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError("pilot custom nominations must be distinct")
    runs = [_validate_run(run, plan) for run in runs]
    by_key = {(run["tokenizer_id"], run["seed"]): run for run in runs}
    if len(by_key) != len(runs):
        raise ValueError("pilot run identities must be unique")
    screen_keys = {(plan["primary_baseline"], 42), *((value, 42) for value in custom_ids)}
    if not screen_keys <= set(by_key):
        raise ValueError("pilot screen matrix is incomplete")
    screen_custom = [by_key[(value, 42)] for value in custom_ids]
    selected = min(
        screen_custom,
        key=lambda run: (
            _macro(run, "fixed_document"),
            run["fixed_document"]["active_seconds"],
            run["vocab_size"],
            run["tokenizer_id"],
        ),
    )["tokenizer_id"]
    expected_keys = (
        screen_keys
        | {(plan["primary_baseline"], seed) for seed in (43, 44)}
        | {(selected, seed) for seed in (43, 44)}
    )
    if set(by_key) != expected_keys or len(runs) != plan["confirmation"]["total_runs"]:
        raise ValueError("pilot confirmation matrix must contain exactly seven frozen runs")
    baseline_runs = [by_key[(plan["primary_baseline"], seed)] for seed in (42, 43, 44)]
    candidate_runs = [by_key[(selected, seed)] for seed in (42, 43, 44)]
    identities = {run["document_stream_sha256"] for run in runs}
    backbone_ids = {run["backbone_manifest_sha256"] for run in runs}
    target_flops = baseline_runs[0]["fixed_document"]["analytic_flops"]
    model_ids = {}
    tokenizer_ids = {}
    for run in runs:
        model_ids.setdefault(run["tokenizer_id"], set()).add(run["model_manifest_sha256"])
        tokenizer_ids.setdefault(run["tokenizer_id"], set()).add(run["tokenizer_model_sha256"])
    if (
        len(identities) != 1
        or len(backbone_ids) != 1
        or any(len(values) != 1 for values in model_ids.values())
        or any(len(values) != 1 for values in tokenizer_ids.values())
        or baseline_runs[0]["fixed_document"]["tokenizer_tokens"]
        != plan["stopping"]["fixed_document_mistral_tokens"]
        or any(
            run["fixed_flop"]["target_analytic_flops"] != target_flops
            or run["fixed_flop"]["analytic_flops"] != target_flops
            for run in runs
        )
    ):
        raise ValueError("pilot model/document identity or fixed-FLOP target is unmatched")
    views = {}
    eligible = True
    seed_deltas = {}
    for view in VIEWS:
        deltas, seed_means = _paired_document_deltas(candidate_runs, baseline_runs, view)
        report = _bootstrap(deltas, plan, view)
        report["aggregate_guardrail_pass"] = (
            report["macro_upper_95_bpb"] <= plan["eligibility"]["aggregate_upper_95_bpb"]
        )
        report["category_guardrails_pass"] = all(
            value["upper_95_bpb"] <= plan["eligibility"]["category_upper_95_bpb"]
            for value in report["categories"].values()
        )
        eligible &= report["aggregate_guardrail_pass"] and report["category_guardrails_pass"]
        views[view] = report
        seed_deltas[view] = seed_means
    quality_target = _macro(baseline_runs[0], "fixed_document")
    baseline_flops = [_flops_to_quality(run, quality_target) for run in baseline_runs]
    candidate_flops = [_flops_to_quality(run, quality_target) for run in candidate_runs]
    baseline_seconds = [_seconds_to_quality(run, quality_target) for run in baseline_runs]
    candidate_seconds = [_seconds_to_quality(run, quality_target) for run in candidate_runs]
    baseline_reaches_quality = all(value is not None for value in baseline_flops)
    reaches_quality = all(value is not None for value in candidate_flops)
    eligible &= baseline_reaches_quality and reaches_quality
    baseline_median = median(baseline_flops) if baseline_reaches_quality else None
    candidate_median = (
        median(value for value in candidate_flops if value is not None) if reaches_quality else None
    )
    baseline_median_seconds = median(baseline_seconds) if baseline_reaches_quality else None
    candidate_median_seconds = (
        median(value for value in candidate_seconds if value is not None)
        if reaches_quality
        else None
    )
    baseline_vocab = by_key[(plan["primary_baseline"], 42)]["vocab_size"]
    candidate_vocab = by_key[(selected, 42)]["vocab_size"]
    custom_first = eligible and (
        candidate_median < baseline_median
        or (candidate_median == baseline_median and candidate_vocab < baseline_vocab)
    )
    ranking = (
        [selected, plan["primary_baseline"]]
        if custom_first
        else [plan["primary_baseline"], selected]
    )
    return {
        "format": RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "fixture_analysis_complete_no_pilot_or_audit_authority"
            if fixture
            else "provisional_ranking_complete_D5_audit_unopened"
        ),
        "screen": {
            "seed": 42,
            "custom_scores": {
                run["tokenizer_id"]: {
                    "fixed_document_macro_bpb": _macro(run, "fixed_document"),
                    "active_seconds": run["fixed_document"]["active_seconds"],
                }
                for run in screen_custom
            },
            "selected_custom": selected,
        },
        "run_identities": [
            {
                "tokenizer_id": run["tokenizer_id"],
                "seed": run["seed"],
                "tokenizer_model_sha256": run["tokenizer_model_sha256"],
                "model_manifest_sha256": run["model_manifest_sha256"],
                "backbone_manifest_sha256": run["backbone_manifest_sha256"],
                "document_stream_sha256": run["document_stream_sha256"],
                "total_parameters": run["total_parameters"],
            }
            for run in sorted(runs, key=lambda value: (value["seed"], value["tokenizer_id"]))
        ],
        "confirmation": {
            "seeds": [42, 43, 44],
            "views": views,
            "seed_deltas": seed_deltas,
            "eligible_custom": eligible,
        },
        "compute_to_quality": {
            "quality_target_bpb": quality_target,
            "baseline_flops_by_seed": baseline_flops,
            "candidate_flops_by_seed": candidate_flops,
            "baseline_median_flops": baseline_median,
            "candidate_median_flops": candidate_median,
            "baseline_seconds_by_seed": baseline_seconds,
            "candidate_seconds_by_seed": candidate_seconds,
            "baseline_median_seconds": baseline_median_seconds,
            "candidate_median_seconds": candidate_median_seconds,
            "candidate_reaches_target_all_seeds": reaches_quality,
            "baseline_reaches_target_all_seeds": baseline_reaches_quality,
        },
        "fixed_wall_clock_secondary": {
            "target_seconds": baseline_runs[0]["fixed_document"]["active_seconds"],
            "baseline_macro_bpb_by_seed": {
                str(run["seed"]): _bpb_at_wallclock(
                    run, baseline_runs[0]["fixed_document"]["active_seconds"]
                )
                for run in baseline_runs
            },
            "candidate_macro_bpb_by_seed": {
                str(run["seed"]): _bpb_at_wallclock(
                    run, baseline_runs[0]["fixed_document"]["active_seconds"]
                )
                for run in candidate_runs
            },
            "selection_stopping_view": False,
        },
        "systems": {
            tokenizer_id: {
                str(run["seed"]): {
                    view: {
                        "active_seconds": run[view]["active_seconds"],
                        "peak_memory_bytes": run[view]["peak_memory_bytes"],
                        "throughput_tokens_per_second": run[view]["throughput_tokens_per_second"],
                    }
                    for view in VIEWS
                }
                for run in runs
                if run["tokenizer_id"] == tokenizer_id
            }
            for tokenizer_id in sorted({run["tokenizer_id"] for run in runs})
        },
        "provisional_ranking": ranking,
        "audit_handoff": {
            "format": "speck_sealed_audit_open_request_draft",
            "audit_identity": "D5_tokenizer",
            "ranked_finalists": ranking,
            "opening_authorized": False,
            "fallback": "mistral-32k",
        },
        "selection_authority": False,
        "fixture": fixture,
    }
