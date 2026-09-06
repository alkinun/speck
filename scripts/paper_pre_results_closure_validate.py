"""Validate the Paper 1 cross-program pre-results closure matrix."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closure", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bound_json(root, reference, message):
    path = root / reference.get("path", "")
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError(message)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_closure(closure, root):
    if (
        closure.get("format") != "speck_paper1_pre_results_closure"
        or closure.get("format_version") != 1
        or closure.get("status")
        != "autonomous_static_work_closed_event_and_external_authority_gates_open"
    ):
        raise ValueError("invalid Paper 1 pre-results closure identity")

    operational = closure.get("operational_chain", {})
    program = bound_json(root, operational.get("active_program", {}), "active program changed")
    for name in (
        "event_runner",
        "automation",
        "live_launch",
        "active_rerun",
        "program_source_gate",
        "postcommit_acceptance",
    ):
        reference = operational.get(name, {})
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError(f"Paper 1 operational chain changed: {name}")
    automation = json.loads(
        (root / operational["automation"]["path"]).read_text(encoding="utf-8")
    )
    live_launch = json.loads(
        (root / operational["live_launch"]["path"]).read_text(encoding="utf-8")
    )
    source_gate = json.loads(
        (root / operational["program_source_gate"]["path"]).read_text(encoding="utf-8")
    )
    acceptance = json.loads(
        (root / operational["postcommit_acceptance"]["path"]).read_text(encoding="utf-8")
    )
    state = operational.get("state", {})
    evidence = program.get("finalist_evidence", {})
    if (
        state.get("language_sequence_terminal") is not False
        or evidence.get("analysis_result") is not None
        or evidence.get("control_results") != []
        or evidence.get("candidate_results") != []
        or state.get("automatic_event_chain_is_only_authorized_training_mutation") is not True
        or state.get("quality_dependent_branching") is not False
        or state.get("automatic_retry") is not False
        or state.get("manual_result_interpretation_before_v3_acceptance") is not False
        or state.get("postcommit_v3_wired_into_frozen_runner") is not False
        or state.get("postcommit_v3_must_run_before_interpretation_or_Linear_completion") is not True
        or automation.get("decision", {}).get("event_driven_no_polling") is not True
        or live_launch.get("decision", {}).get("initial_control_launch_authorized") is not True
        or source_gate.get("decision", {}).get(
            "program_validator_source_gate_integration_qualified"
        )
        is not True
        or acceptance.get("decision", {}).get("wired_into_frozen_automation") is not False
        or acceptance.get("decision", {}).get("must_run_after_every_automatic_result_commit") is not True
        or acceptance.get("decision", {}).get("must_pass_before_result_interpretation_or_linear_completion")
        is not True
    ):
        raise ValueError("Paper 1 operational state or acceptance boundary changed")

    tracks = closure.get("closed_autonomous_tracks", {})
    if set(tracks) != {"sequence", "depth", "width", "systems"}:
        raise ValueError("Paper 1 closed autonomous tracks changed")
    sequence = bound_json(root, tracks["sequence"]["qualification"], "sequence closure changed")
    depth = bound_json(root, tracks["depth"]["qualification"], "depth closure changed")
    width = bound_json(root, tracks["width"]["qualification"], "width closure changed")
    systems = bound_json(root, tracks["systems"]["audit"], "systems closure changed")
    if (
        sequence.get("decision", {}).get("pre_results_sequence_source_and_factorization_work_complete")
        is not True
        or sequence.get("decision", {}).get("training_authorized") is not False
        or depth.get("decision", {}).get("sequence_parent_selected") is not False
        or depth.get("decision", {}).get("training_authorized") is not False
        or width.get("decision", {}).get("pre_results_width_reference_work_complete") is not True
        or width.get("decision", {}).get("training_authorized") is not False
        or len(systems.get("remaining_activation_gates", [])) != 5
        or any(gate.get("current_pass") is not False for gate in systems["remaining_activation_gates"])
        or systems.get("decision", {}).get("further_live_work_during_language_sequence_authorized")
        is not False
        or systems.get("decision", {}).get("systems_execution_authorized") is not False
        or tracks["systems"].get("systems_execution_authorized") is not False
        or any(track.get("implementation_authorized") is not False for track in tracks.values() if "implementation_authorized" in track)
        or any(track.get("training_authorized") is not False for track in tracks.values() if "training_authorized" in track)
    ):
        raise ValueError("Paper 1 autonomous track closure changed")

    downstream = closure.get("downstream_architecture_gates", {})
    interaction = bound_json(
        root, downstream.get("interaction", {}).get("qualification", {}), "interaction gate changed"
    )
    scaling = bound_json(root, downstream.get("scaling", {}).get("design", {}), "scaling gate changed")
    if (
        interaction.get("resource_and_authority", {}).get("axis_bundles_selected") is not False
        or interaction.get("resource_and_authority", {}).get("cube_materialized") is not False
        or interaction.get("decision", {}).get("training_authorized") is not False
        or scaling.get("decision", {}).get("candidate_architecture_selected") is not False
        or scaling.get("decision", {}).get("fit_runs_authorized") is not False
        or scaling.get("decision", {}).get("paper_scale_authorized") is not False
    ):
        raise ValueError("Paper 1 downstream architecture gate changed")

    external = closure.get("external_authority_gates", {})
    manifest = bound_json(root, external.get("evaluation_manifest", {}), "evaluation manifest changed")
    if manifest.get("status") != external.get("evaluation_manifest", {}).get("status"):
        raise ValueError("Paper 1 evaluation status changed")
    helmet = external.get("HELMET", {})
    helmet_values = {
        name: bound_json(root, reference, f"HELMET gate changed: {name}")
        for name, reference in helmet.items()
        if isinstance(reference, dict) and "path" in reference
    }
    if (
        set(helmet_values)
        != {
            "archive_rights",
            "synthetic_reconstruction",
            "real_data_reconstruction",
            "truncation_tokenizer",
            "model_judge",
        }
        or helmet.get("local_next_action_available") is not False
        or helmet.get("extraction_or_execution_authorized") is not False
        or len(helmet.get("requires", [])) != 3
        or helmet_values["archive_rights"].get("decision", {}).get("extraction_authorized")
        is not False
        or helmet_values["synthetic_reconstruction"].get("decision", {}).get(
            "official_helmet_recall_reconstruction_qualified"
        )
        is not False
        or helmet_values["real_data_reconstruction"].get("decision", {}).get("rights_qualified")
        is not False
        or helmet_values["truncation_tokenizer"].get("decision", {}).get(
            "license_acceptance_authorized"
        )
        is not False
        or helmet_values["model_judge"].get("decision", {}).get("data_transmission_authorized")
        is not False
    ):
        raise ValueError("Paper 1 HELMET external-authority boundary changed")
    novelty_reference = external.get("novelty", {})
    novelty = bound_json(root, novelty_reference.get("packet", {}), "novelty review packet changed")
    if (
        novelty_reference.get("local_next_action_available") is not False
        or "independent" not in novelty_reference.get("requires", "")
        or novelty_reference.get("experiment_authorized") is not False
        or novelty.get("decision", {}).get("independent_review_completed") is not False
        or novelty.get("decision", {}).get("n1_experiment_authorized") is not False
    ):
        raise ValueError("Paper 1 novelty external-authority boundary changed")

    allowed = closure.get("currently_allowed_actions", [])
    forbidden = closure.get("currently_forbidden_actions", [])
    if (
        len(allowed) != 6
        or len(forbidden) != 8
        or not any("standalone result-acceptance v3" in item for item in allowed)
        or not any("infrequent handle-specific" in item for item in allowed)
        or not any("logs, metrics, checkpoint tensors" in item for item in forbidden)
        or not any("sequence DAG descendant" in item for item in forbidden)
        or not any("Speck's own review" in item for item in forbidden)
    ):
        raise ValueError("Paper 1 allowed/forbidden action boundary changed")
    decision = closure.get("decision", {})
    if (
        decision.get("autonomous_static_research_exhausted_without_named_new_evidence") is not True
        or decision.get("active_event_chain_remains_progress_path") is not True
        or decision.get("overall_goal_complete") is not False
        or decision.get("blocked_goal_status_appropriate") is not False
        or "authorized event-driven training handle" not in decision.get("reason_not_blocked", "")
        or decision.get("implementation_authorized") is not False
        or decision.get("new_training_authorized_outside_event_chain") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("Paper 1 pre-results closure decision changed")
    return {
        "status": "valid_event_wait_required",
        "closed_autonomous_tracks": len(tracks),
        "systems_live_gates": len(systems["remaining_activation_gates"]),
        "external_authority_domains": 2,
        "allowed_actions": len(allowed),
        "goal_complete": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_closure(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).closure)
    print(
        "Paper 1 pre-results closure: "
        f"{report['status']} ({report['closed_autonomous_tracks']} tracks, "
        f"{report['systems_live_gates']} systems gates)"
    )


if __name__ == "__main__":
    main()
