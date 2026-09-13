"""Optional disjoint phase and checkpoint accounting for production preprocessing."""

import time


class PreprocessTiming:
    def __init__(self, report):
        if report is not None and (not isinstance(report, dict) or report):
            raise ValueError("preprocess timing requires a fresh report dictionary")
        self.report = report
        self.started = self.mark = time.perf_counter()
        self.current_phase = "setup"
        self.checkpoint_seconds = 0.0
        if report is not None:
            report.update(
                {
                    "format": "speck_preprocess_timing",
                    "format_version": 1,
                    "status": "incomplete",
                    "phases": {},
                    "sources": [],
                }
            )

    def phase(self, name):
        if self.report is None:
            return
        now = time.perf_counter()
        phases = self.report["phases"]
        phases[self.current_phase] = phases.get(self.current_phase, 0.0) + now - self.mark
        self.current_phase, self.mark = name, now

    def start_source(self, source_id, counts):
        if self.report is None:
            return
        self.source_id = source_id
        self.source_started = time.perf_counter()
        self.source_checkpoints = self.checkpoint_seconds
        self.source_seen = counts.get("records_seen", 0)
        self.source_retained = counts.get("records_retained", 0)

    def end_source(self, counts, utf8_bytes):
        if self.report is None:
            return
        elapsed = time.perf_counter() - self.source_started
        checkpoints = self.checkpoint_seconds - self.source_checkpoints
        self.report["sources"].append(
            {
                "id": self.source_id,
                "records_seen": counts.get("records_seen", 0) - self.source_seen,
                "records_retained": counts.get("records_retained", 0) - self.source_retained,
                "processed_utf8_bytes": utf8_bytes,
                "elapsed_seconds": elapsed,
                "checkpoint_seconds": checkpoints,
                "processing_seconds_excluding_checkpoints": elapsed - checkpoints,
            }
        )

    def checkpoint(self, function, *args):
        if self.report is None:
            return function(*args)
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            self.checkpoint_seconds += time.perf_counter() - started

    def finish(self, status):
        if self.report is None:
            return
        self.phase("finished")
        self.report.update(
            {
                "status": status,
                "total_seconds": self.mark - self.started,
                "checkpoint_seconds_all_phases": self.checkpoint_seconds,
                "boundary": "Disjoint wall-time phases; source elapsed includes source checkpoints. All-phase checkpoint time overlaps phases and must not be added again. Processing excludes checkpoints but includes reads, parsing, hashing, candidate comparisons and index updates; it is not a kernel-only or production-scale steady-state benchmark.",
            }
        )
