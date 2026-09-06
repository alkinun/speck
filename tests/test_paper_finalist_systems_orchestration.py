from pathlib import Path

from speck.paper_finalist_systems_orchestration import orchestrate_block
from speck.paper_finalist_systems_telemetry import GPU_UUID
from speck.paper_finalist_systems_workload import build_trial_plan

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
qualification_path = root / "results" / "Speck-Paper1" / "finalist-qualification-v1.json"


class RecordingAdapter:
    def __init__(self, *, temperatures=(44, 44), fail_engine=None, idle_delta=0):
        self.time_ns = 0
        self.calls = []
        self.temperatures = temperatures
        self.fail_engine = fail_engine
        self.idle_delta = idle_delta
        self.gate_index = 0
        self.failures = []

    def now_ns(self):
        return self.time_ns

    def start_sampler(self, block):
        self.calls.append(("start_sampler", block))
        return {"block": block}

    def idle(self, sampler, seconds, phase):
        self.calls.append(("idle", phase, seconds))
        start = self.time_ns
        self.time_ns += (seconds + self.idle_delta) * 1_000_000_000
        return {"start_ns": start, "end_ns": self.time_ns}

    def recover(self, sampler, maximum_temperature, timeout):
        self.calls.append(("recover", maximum_temperature, timeout))
        self.time_ns += 30 * 1_000_000_000
        return {"elapsed_seconds": 30, "end_temperature_c": 44}

    def live_gate(self, trial):
        self.calls.append(("gate", trial["position"], trial["role"]))
        temperature = self.temperatures[self.gate_index]
        self.gate_index += 1
        return {
            "gpu_uuid": GPU_UUID,
            "temperature_c": temperature,
            "gpu_utilization_percent": 0,
            "compute_process_pids": [],
        }

    def run_engine(self, trial, sampler):
        self.calls.append(("engine", trial["position"], trial["role"]))
        if self.fail_engine == trial["position"]:
            raise RuntimeError("engine failure")
        self.time_ns += 10 * 1_000_000_000
        return {"position": trial["position"], "role": trial["role"]}

    def stop_sampler(self, sampler):
        self.calls.append(("stop_sampler", sampler["block"]))
        return {"samples": "mocked"}

    def build_traces(self, block, samples, pre_idle, post_idle, engines, gates):
        self.calls.append(("build_traces", block))
        return [{"role": engine["role"]} for engine in engines]

    def assemble_block(self, block, trials, engines, traces):
        self.calls.append(("assemble_block", block))
        return {"status": "complete_qualified", "block": block}

    def retain_failure(self, failure):
        self.calls.append(("retain_failure", failure["block"]))
        self.failures.append(failure)
        return {"path": "failure.json", "sha256": "a" * 64}


def plan(tmp_path):
    return build_trial_plan(protocol_path, qualification_path, tmp_path / "outputs")


def test_orchestration_runs_one_continuous_sampler_and_frozen_event_order(tmp_path):
    adapter = RecordingAdapter()
    result = orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    assert result["status"] == "complete_qualified"
    assert result["events"] == [
        "sampler_start",
        "pre_block_idle",
        "trial_0_gate",
        "trial_0_engine",
        "thermal_recovery",
        "trial_1_gate",
        "trial_1_engine",
        "post_block_idle",
        "sampler_stop",
        "trace_build",
        "block_assembly",
    ]
    assert sum(call[0] == "start_sampler" for call in adapter.calls) == 1
    assert sum(call[0] == "stop_sampler" for call in adapter.calls) == 1
    assert result["pre_idle"]["end_ns"] - result["pre_idle"]["start_ns"] == 120e9
    assert result["post_idle"]["end_ns"] - result["post_idle"]["start_ns"] == 120e9
    assert result["replacement_or_retry_authorized"] is False
    assert result["successor_block_authorized"] is True


def test_orchestration_stops_sampler_and_retains_first_engine_failure(tmp_path):
    adapter = RecordingAdapter(fail_engine=0)
    result = orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    assert result["status"] == "failed_retained"
    assert result["failure"] == "engine failure"
    assert "sampler_stop_after_failure" in result["events"]
    assert not any(call[:2] == ("gate", 1) for call in adapter.calls)
    assert result["replacement_or_retry_authorized"] is False
    assert result["successor_block_authorized"] is False


def test_orchestration_rejects_hot_trial_gate_before_engine(tmp_path):
    adapter = RecordingAdapter(temperatures=(46, 44))
    result = orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    assert result["status"] == "failed_retained"
    assert "live gate failed" in result["failure"]
    assert not any(call[0] == "engine" for call in adapter.calls)


def test_orchestration_rejects_paired_temperature_difference(tmp_path):
    adapter = RecordingAdapter(temperatures=(45, 42))
    result = orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    assert result["status"] == "failed_retained"
    assert "paired start temperatures differ" in result["failure"]
    assert sum(call[0] == "engine" for call in adapter.calls) == 1
    assert not any(call[:2] == ("idle", "post_block") for call in adapter.calls)


def test_orchestration_rejects_changed_idle_duration(tmp_path):
    adapter = RecordingAdapter(idle_delta=1)
    result = orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    assert result["status"] == "failed_retained"
    assert "idle interval changed" in result["failure"]
    assert sum(call[0] == "stop_sampler" for call in adapter.calls) == 1


def test_orchestration_rejects_exhausted_budget_before_sampler(tmp_path):
    adapter = RecordingAdapter()
    adapter.time_ns = 4 * 3600 * 1_000_000_000
    try:
        orchestrate_block(protocol_path, plan(tmp_path), 0, adapter, 0)
    except ValueError as error:
        assert "budget is exhausted" in str(error)
    else:
        raise AssertionError("exhausted budget was accepted")
    assert adapter.calls == []


def test_final_block_has_no_successor_authority(tmp_path):
    adapter = RecordingAdapter()
    result = orchestrate_block(protocol_path, plan(tmp_path), 5, adapter, 0)
    assert result["status"] == "complete_qualified"
    assert result["successor_block_authorized"] is False
