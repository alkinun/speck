import json

from scripts.moe_screen_analyze import analyze, assess_arm, select_arm

ANALYSIS = {
    "single_seed_tie_margin_nats": 0.01,
    "stability_window_starts_at_tokens": 50,
    "stability_limits": {
        "all_values_finite": True,
        "maximum_normalized_utilization_cv": 0.5,
        "maximum_zero_load_experts": 0,
        "minimum_normalized_entropy": 0.25,
    },
}
ARM = {"parameters": 100, "active_parameters": 50, "router_parameters": 10}


def summary(loss=2.0, entropy=0.8, cv=0.2, zero=0):
    return {
        "partial": False,
        "steps": 10,
        "completed_steps": 10,
        "validation_history": [{"step": 10, "validation_loss": loss}],
        "routing_history": [
            {
                "training_tokens": 50,
                "layers": [
                    {
                        "layer": "layer-0",
                        "normalized_entropy": entropy,
                        "utilization_min": 0.1,
                        "utilization_max": 0.4,
                        "utilization_cv": cv,
                        "zero_load_experts": zero,
                        "logit_rms": 1.2,
                        "selection_bias_rms": 0.0,
                    }
                ],
            }
        ],
    }


def test_assess_arm_applies_frozen_stability_limits():
    assert assess_arm("good", ARM, summary(), ANALYSIS)["eligible"]
    bad = assess_arm("bad", ARM, summary(entropy=0.1, cv=0.7, zero=1), ANALYSIS)
    assert not bad["eligible"]
    assert len(bad["ineligibility_reasons"]) == 3


def test_select_arm_uses_preference_only_inside_tie_margin():
    results = {
        "g8": {"name": "g8", "eligible": True, "final_validation_loss": 2.005},
        "g16": {"name": "g16", "eligible": True, "final_validation_loss": 2.0},
        "g32": {"name": "g32", "eligible": True, "final_validation_loss": 2.02},
    }
    decision = select_arm(["g8", "g16", "g32"], results, 0.01, ["g8", "g16", "g32"])
    assert decision["selected"] == "g8"
    assert decision["ranked"] == ["g16", "g8", "g32"]


def test_analyze_never_selects_dense_versus_moe(tmp_path):
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    contract = {
        "format": "speck_moe_design_screen",
        "format_version": 2,
        "analysis": {
            **ANALYSIS,
            "tie_break_order": {
                "granularity": ["g8", "g16", "g32"],
                "placement_capacity": [
                    "p-shared",
                    "g8",
                    "p-dense-first",
                    "p-interleaved",
                ],
                "overall_moe_design": [
                    "p-shared",
                    "g8",
                    "g16",
                    "g32",
                    "p-dense-first",
                    "p-interleaved",
                ],
            },
        },
        "arms": {
            name: {**ARM, "router_parameters": 0 if name == "dense" else 10}
            for name in (
                "dense",
                "g8",
                "g16",
                "g32",
                "p-dense-first",
                "p-interleaved",
                "p-shared",
            )
        },
    }
    (screen_dir / "screen.json").write_text(json.dumps(contract), encoding="utf-8")
    root = tmp_path / "checkpoints"
    for name in contract["arms"]:
        directory = root / f"SpeckLC-150M-MoEScreen-{name}"
        directory.mkdir(parents=True)
        value = summary(loss=2.0)
        if name == "dense":
            value["routing_history"] = []
        (directory / "run_summary.json").write_text(json.dumps(value), encoding="utf-8")

    report = analyze(screen_dir / "screen.json", root)

    assert report["decisions"]["dense_versus_moe"]["selected"] is None
    assert report["decisions"]["granularity"]["selected"] == "g8"
    assert report["decisions"]["placement_capacity"]["selected"] == "p-shared"
    assert report["decisions"]["overall_moe_design"]["selected"] == "p-shared"
