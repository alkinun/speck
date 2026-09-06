"""Exhaustively qualify the isolated Stable LatentMoE CPU references."""

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from speck.stable_latentmoe_reference import (
    exact_quantile_bias,
    histogram_quantile_bias,
    histogram_quantile_counts,
    normalized_latent_moe,
    qb_required_bias,
    quantile_routes,
    rms_norm,
    situ_glu,
)

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_revision():
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("Stable LatentMoE qualification requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_protocol(protocol, root=ROOT):
    if (
        protocol.get("format") != "speck_stable_latentmoe_CPU_reference_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("status") != "frozen_after_source_gate_before_qualification"
    ):
        raise ValueError("invalid Stable LatentMoE CPU reference protocol identity")
    for reference in (*protocol.get("authority", {}).values(), protocol.get("implementation", {})):
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError("Stable LatentMoE CPU reference input changed")
    tests = protocol.get("unit_tests", {})
    if not (root / tests.get("path", "")).is_file() or file_sha256(
        root / tests.get("path", "")
    ) != tests.get("sha256"):
        raise ValueError("Stable LatentMoE CPU reference tests changed")
    primitives = protocol.get("primitives", {})
    matrix = protocol.get("qualification_matrix", {})
    rule = protocol.get("pass_rule", {})
    boundary = protocol.get("boundary", {})
    if (
        set(primitives)
        != {
            "SiTU_GLU",
            "normalized_LatentMoE",
            "exact_Quantile_Balancing",
            "histogram_Quantile_Balancing",
        }
        or primitives["SiTU_GLU"].get("source_equation") != 12
        or primitives["normalized_LatentMoE"].get("source_equation") != 11
        or primitives["exact_Quantile_Balancing"].get("source_equations") != [13, 14]
        or primitives["histogram_Quantile_Balancing"].get("source_bins") != 1000
        or primitives["histogram_Quantile_Balancing"].get("EMA_included") is not False
        or matrix.get("dtype") != "torch.float64"
        or matrix.get("Quantile_Balancing", {}).get("required_production_fidelity_bins") != 1000
        or rule.get("maximum_manual_forward_error_at_most") != 1e-12
        or rule.get("maximum_histogram_interval_error_in_bin_widths_at_most") != 1.0
        or rule.get("partition_count_mismatches") != 0
        or boundary.get("architecture_or_model_code_changed") is not False
        or boundary.get("active_experiment_program_changed") is not False
        or boundary.get("active_result_checkpoint_service_GPU_or_runtime_accessed") is not False
        or boundary.get("conventional_MoE_qualified") is not False
        or boundary.get("primitive_composition_authorized") is not False
        or boundary.get("model_integration_authorized") is not False
        or boundary.get("training_authorized") is not False
        or boundary.get("promotion_authority") is not False
    ):
        raise ValueError("Stable LatentMoE CPU qualification contract changed")
    return matrix


def qualify_situ(config):
    grid = torch.linspace(
        config["scalar_grid_min"],
        config["scalar_grid_max"],
        config["scalar_grid_points"],
        dtype=torch.float64,
    )
    maximum_formula_error = 0.0
    maximum_bound_ratio = 0.0
    finite_gradient_cases = 0
    cases = 0
    for gate_beta, up_beta in config["beta_pairs"]:
        gate = grid.clone().requires_grad_()
        up = grid.flip(0).clone().requires_grad_()
        value = situ_glu(gate, up, gate_beta, up_beta)
        manual = gate_beta * torch.tanh(gate / gate_beta) * torch.sigmoid(gate)
        manual = manual * up_beta * torch.tanh(up / up_beta)
        maximum_formula_error = max(maximum_formula_error, (value - manual).abs().max().item())
        maximum_bound_ratio = max(
            maximum_bound_ratio, value.abs().max().item() / (gate_beta * up_beta)
        )
        value.sum().backward()
        if not torch.isfinite(gate.grad).all() or not torch.isfinite(up.grad).all():
            raise RuntimeError("SiTU produced a non-finite gradient")
        finite_gradient_cases += 1
        cases += value.numel()
    for seed in range(config["random_seeds"]):
        generator = torch.Generator().manual_seed(50_000 + seed)
        gate = (torch.randn((7, 11), generator=generator, dtype=torch.float64) * 50).requires_grad_()
        up = (torch.randn((7, 11), generator=generator, dtype=torch.float64) * 50).requires_grad_()
        value = situ_glu(gate, up)
        if value.abs().max().item() > 100:
            raise RuntimeError("SiTU violated its source coordinate bound")
        value.square().mean().backward()
        if not torch.isfinite(gate.grad).all() or not torch.isfinite(up.grad).all():
            raise RuntimeError("SiTU random case produced a non-finite gradient")
        finite_gradient_cases += 1
        cases += value.numel()
    scale = config["small_scale"]
    small_gate = torch.tensor([-2, -1, 1, 2], dtype=torch.float64) * scale
    small_up = torch.tensor([2, -1, -2, 1], dtype=torch.float64) * scale
    local_error = (situ_glu(small_gate, small_up) - F.silu(small_gate) * small_up).abs().max().item()
    large_beta = config["large_beta"]
    limit_error = (
        situ_glu(grid, grid.flip(0), large_beta, large_beta) - F.silu(grid) * grid.flip(0)
    ).abs().max().item()
    if maximum_formula_error > 1e-12 or maximum_bound_ratio > 1 or not math.isfinite(limit_error):
        raise RuntimeError("SiTU formula, bound, or limit gate failed")
    return {
        "scalar_values": cases,
        "finite_gradient_cases": finite_gradient_cases,
        "maximum_formula_error": maximum_formula_error,
        "maximum_coordinate_bound_ratio": maximum_bound_ratio,
        "small_scale_SwiGLU_absolute_error": local_error,
        "large_beta_SwiGLU_absolute_error": limit_error,
    }


def _random_linear(generator, output, input_):
    return torch.randn((output, input_), generator=generator, dtype=torch.float64, requires_grad=True)


def qualify_latent(config):
    maximum_forward_error = 0.0
    cases = 0
    gradient_tensors = 0
    shapes = itertools.product(
        config["token_counts"],
        config["full_widths"],
        config["latent_widths"],
        config["routed_expert_counts"],
        config["selected_expert_counts"],
        config["shared_expert_counts"],
    )
    valid_shapes = [shape for shape in shapes if shape[4] <= shape[3]]
    for seed in range(config["random_seeds"]):
        for shape_id, (tokens, width, latent, routed_count, selected_count, shared_count) in enumerate(
            valid_shapes
        ):
            generator = torch.Generator().manual_seed(100_000 * seed + shape_id)
            x = torch.randn((tokens, width), generator=generator, dtype=torch.float64, requires_grad=True)
            down = _random_linear(generator, latent, width)
            up = _random_linear(generator, width, latent)
            norm_weight = torch.randn(latent, generator=generator, dtype=torch.float64, requires_grad=True)
            routed_matrices = [
                _random_linear(generator, latent, latent) for _ in range(routed_count)
            ]
            shared_matrices = [_random_linear(generator, width, width) for _ in range(shared_count)]
            indices = torch.stack(
                [torch.randperm(routed_count, generator=generator)[:selected_count] for _ in range(tokens)]
            )
            weights = torch.rand((tokens, selected_count), generator=generator, dtype=torch.float64)
            weights /= weights.sum(dim=-1, keepdim=True)
            routed = [
                lambda value, matrix=matrix: F.linear(value, matrix)
                for matrix in routed_matrices
            ]
            shared = [
                lambda value, matrix=matrix: F.linear(value, matrix)
                for matrix in shared_matrices
            ]
            result = normalized_latent_moe(
                x,
                down,
                up,
                routed,
                shared,
                indices,
                weights,
                norm_weight=norm_weight,
                eps=1e-12,
            )
            latent_input = F.linear(x, down)
            routed_all = torch.stack(
                [F.linear(latent_input, matrix) for matrix in routed_matrices], dim=1
            )
            selected = torch.gather(
                routed_all,
                1,
                indices.unsqueeze(-1).expand(-1, -1, latent),
            )
            aggregate = (selected * weights.unsqueeze(-1)).sum(dim=1)
            manual = F.linear(rms_norm(aggregate, norm_weight, 1e-12), up)
            manual += sum(F.linear(x, matrix) for matrix in shared_matrices)
            error = (result["output"] - manual).abs().max().item()
            maximum_forward_error = max(maximum_forward_error, error)
            result["output"].square().mean().backward()
            parameters = [x, down, up, norm_weight, *routed_matrices, *shared_matrices]
            if any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters):
                raise RuntimeError("normalized LatentMoE produced a missing or non-finite gradient")
            gradient_tensors += len(parameters)
            cases += 1
    if maximum_forward_error > config["forward_absolute_tolerance"]:
        raise RuntimeError("normalized LatentMoE manual forward parity failed")
    return {
        "shape_seed_cases": cases,
        "valid_shape_combinations": len(valid_shapes),
        "finite_gradient_tensors": gradient_tensors,
        "maximum_manual_forward_error": maximum_forward_error,
    }


def _uneven_lengths(tokens):
    first = max(1, tokens // 7)
    second = max(1, tokens // 3)
    return [first, second, tokens - first - second]


def _interval_error(value, lower, upper):
    return torch.maximum(torch.maximum(lower - value, value - upper), torch.zeros_like(value))


def qualify_quantile(config):
    exact_cases = 0
    no_boundary_tie_cases = 0
    boundary_tie_experts = 0
    rank_bracket_failures = 0
    partition_cases = 0
    partition_count_mismatches = 0
    histogram_cases = 0
    source_1000_bin_cases = 0
    maximum_interval_error_bin_widths = 0.0
    for shape_id, shape in enumerate(config["shapes"]):
        tokens, experts, selected = shape["tokens"], shape["experts"], shape["selected"]
        target = tokens * selected // experts
        for seed in range(config["random_seeds_per_shape"]):
            generator = torch.Generator().manual_seed(200_000 * shape_id + seed)
            scores = torch.rand((tokens, experts), generator=generator, dtype=torch.float64)
            bias = torch.randn(experts, generator=generator, dtype=torch.float64) * 0.15
            bias -= bias.mean()
            routes = quantile_routes(scores, bias, selected)
            gathered_raw = torch.gather(scores, 1, routes["indices"])
            if not torch.allclose(
                routes["weights"], gathered_raw / gathered_raw.sum(dim=1, keepdim=True), rtol=0, atol=0
            ):
                raise RuntimeError("QB mixture weights include something other than raw scores")
            result = exact_quantile_bias(scores, bias, selected)
            sorted_required = torch.sort(result["required_bias"], dim=0, stable=True).values
            if not torch.equal(result["uncentered_bias"], sorted_required[target]):
                raise RuntimeError("exact QB did not select its source order statistic")
            below = (result["required_bias"] < result["uncentered_bias"]).sum(dim=0)
            below_or_equal = (result["required_bias"] <= result["uncentered_bias"]).sum(dim=0)
            bracket = (below <= target) & (below_or_equal > target)
            rank_bracket_failures += int((~bracket).sum().item())
            ties = below_or_equal - below > 1
            boundary_tie_experts += int(ties.sum().item())
            if not ties.any().item():
                if not torch.equal(below, torch.full_like(below, target)):
                    raise RuntimeError("tie-free exact QB does not leave target values below threshold")
                no_boundary_tie_cases += 1
            shifted = exact_quantile_bias(scores, bias + 2.75, selected)
            if not torch.allclose(result["bias"], shifted["bias"], rtol=0, atol=2e-15):
                raise RuntimeError("mean-centered QB changed under a common old-bias offset")
            if result["bias"].mean().abs().item() > 2e-16:
                raise RuntimeError("exact QB did not mean-center its returned bias")
            exact_cases += 1

            required = qb_required_bias(scores, bias, selected)
            for bins in config["histogram_bins"]:
                pooled = histogram_quantile_counts(required, bias, bins)
                partitions = [
                    list(required.split(1)),
                    list(required.split(_uneven_lengths(tokens))),
                ]
                for parts in partitions:
                    counts = torch.zeros_like(pooled["counts"])
                    for part in parts:
                        counts += histogram_quantile_counts(part, bias, bins)["counts"]
                    partition_count_mismatches += int(not torch.equal(counts, pooled["counts"]))
                    partition_cases += 1
                estimate = histogram_quantile_bias(pooled["counts"], bias, selected)
                lower = sorted_required[target - 1]
                upper = sorted_required[target]
                error = _interval_error(estimate["uncentered_bias"], lower, upper)
                ratio = (error / estimate["bin_width"]).max().item()
                maximum_interval_error_bin_widths = max(maximum_interval_error_bin_widths, ratio)
                if estimate["bias"].mean().abs().item() > 2e-16:
                    raise RuntimeError("histogram QB did not mean-center its returned bias")
                histogram_cases += 1
                source_1000_bin_cases += int(bins == config["required_production_fidelity_bins"])

    adversarial_required = torch.tensor(
        [[-0.91, -0.83], [-0.74, -0.62], [0.52, 0.41], [0.88, 0.79]],
        dtype=torch.float64,
    )
    adversarial_bias = torch.zeros(2, dtype=torch.float64)
    adversarial_histogram = histogram_quantile_counts(adversarial_required, adversarial_bias, 20)
    pooled = histogram_quantile_bias(adversarial_histogram["counts"], adversarial_bias, 1)
    shard_estimates = []
    for shard in adversarial_required.split(2):
        histogram = histogram_quantile_counts(shard, adversarial_bias, 20)
        shard_estimates.append(histogram_quantile_bias(histogram["counts"], adversarial_bias, 1))
    average_shard = torch.stack([value["uncentered_bias"] for value in shard_estimates]).mean(dim=0)
    pooled_not_average_shard = not torch.allclose(
        pooled["uncentered_bias"], average_shard, rtol=0, atol=1e-12
    )
    if (
        rank_bracket_failures
        or partition_count_mismatches
        or maximum_interval_error_bin_widths > 1 + 1e-12
        or not source_1000_bin_cases
        or not pooled_not_average_shard
    ):
        raise RuntimeError("exact or histogram Quantile Balancing qualification failed")
    return {
        "exact_shape_seed_cases": exact_cases,
        "tie_free_complete_cases": no_boundary_tie_cases,
        "boundary_tie_experts": boundary_tie_experts,
        "target_rank_bracket_failures": rank_bracket_failures,
        "histogram_cases": histogram_cases,
        "source_1000_bin_cases": source_1000_bin_cases,
        "partition_cases": partition_cases,
        "partition_count_mismatches": partition_count_mismatches,
        "maximum_interval_error_in_bin_widths": maximum_interval_error_bin_widths,
        "pooled_estimate_differs_from_average_shard_fixture": pooled_not_average_shard,
    }


def qualify(protocol_path):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_object(protocol_path)
    matrix = validate_protocol(protocol)
    revision = repository_revision()
    situ = qualify_situ(matrix["SiTU"])
    latent = qualify_latent(matrix["normalized_LatentMoE"])
    quantile = qualify_quantile(matrix["Quantile_Balancing"])
    rule = protocol["pass_rule"]
    if (
        latent["maximum_manual_forward_error"] > rule["maximum_manual_forward_error_at_most"]
        or quantile["maximum_interval_error_in_bin_widths"]
        > rule["maximum_histogram_interval_error_in_bin_widths_at_most"] + 1e-12
        or quantile["partition_count_mismatches"] != rule["partition_count_mismatches"]
    ):
        raise RuntimeError("Stable LatentMoE CPU reference pass rule failed")
    return {
        "format": "speck_stable_latentmoe_CPU_reference_qualification",
        "format_version": 1,
        "status": "four_isolated_CPU_primitives_qualified_composition_training_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "isolated float64 CPU equation references only; no conventional MoE, model integration, active experiment access, GPU, training, systems, or promotion claim",
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(protocol_path),
        },
        "implementation": protocol["implementation"],
        "unit_tests": protocol["unit_tests"],
        "results": {
            "SiTU_GLU": situ,
            "normalized_LatentMoE": latent,
            "Quantile_Balancing": quantile,
        },
        "decision": {
            "SiTU_GLU_CPU_reference_qualified": True,
            "normalized_LatentMoE_CPU_reference_qualified": True,
            "exact_Quantile_Balancing_CPU_reference_qualified": True,
            "histogram_Quantile_Balancing_CPU_reference_qualified": True,
            "source_1000_bin_histogram_qualified": True,
            "cutoff_ties_observed_and_rank_bracketed": True,
            "primitive_composition_authorized": False,
            "conventional_MoE_qualified": False,
            "model_integration_authorized": False,
            "training_authorized": False,
            "promotion_authority": False,
            "next_action": "freeze the qualification artifact and retain it behind conventional-MoE, parent, resource, intervention, hardware, and active-finalist gates",
        },
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def main(argv=None):
    args = arguments(argv)
    result = qualify(args.protocol)
    atomic_json(args.output, result)
    print(
        "Stable LatentMoE CPU references: "
        f"{result['status']} "
        f"({result['results']['normalized_LatentMoE']['shape_seed_cases']} latent cases, "
        f"{result['results']['Quantile_Balancing']['histogram_cases']} histogram cases)"
    )


if __name__ == "__main__":
    main()
