"""Supervise finite worker processes under one deadline, preserving their logs on failure."""

import os
import signal
import subprocess
import time

from speck.provenance.io import durable_json


def supervise(command, directory, timeout_seconds, grace_seconds, worker_count=1):
    """Own each rank's process group directly; torchrun ranks create independent sessions."""
    started = time.monotonic()
    processes = []
    reason, error = "exited", None
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    def send_all(sig):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (directory / "worker.log").open("xb") as log:
            for rank in range(worker_count):
                environment = {
                    **os.environ,
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(worker_count),
                }
                processes.append(
                    subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                durable_json(
                    directory / "processes.json",
                    {
                        "supervisor_pid": os.getpid(),
                        "ranks": [
                            {"rank": i, "pid": p.pid, "process_group": p.pid}
                            for i, p in enumerate(processes)
                        ],
                        "boundary": "Diagnostic process IDs only; check current owner, command, allocation and process start time before recovery because IDs can be reused.",
                    },
                )
            while True:
                codes = [p.poll() for p in processes]
                if any(c is not None and c != 0 for c in codes):
                    reason = "worker_failure"
                    break
                if all(c is not None for c in codes):
                    break
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    reason = "timeout"
                    break
                time.sleep(min(0.05, remaining))
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exception:
        reason, error = "launch_failure", repr(exception)
    finally:
        # Repeated cancellation must not interrupt the cleanup that stops the rank groups.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Terminate every known rank group, including descendants whose rank leader already exited.
        send_all(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        send_all(signal.SIGKILL)
        for process in processes:
            process.wait()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    codes = [p.returncode for p in processes]
    return {
        "termination": reason,
        "returncode": next((c for c in codes if c != 0), 0) if len(codes) == worker_count else None,
        "rank_returncodes": codes,
        "supervised_wall_seconds": time.monotonic() - started,
        "launch_error": error,
    }
