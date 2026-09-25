"""Run frozen Python tests in a fail-closed Linux namespace sandbox."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER = """import contextlib, json, os, resource
resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
job = json.load(open('/job.json'))
report = {'status': 'failed'}
try:
    program = compile(job['code'], '<candidate>', 'exec')
    with open(os.devnull, 'w') as sink, contextlib.redirect_stderr(sink):
        namespace = {'__name__': '__main__'}
        with contextlib.redirect_stdout(sink):
            exec(program, namespace)
            exec(compile(job['test'], '<frozen-test>', 'exec'), namespace)
            namespace['check'](namespace[job['entry_point']])
    report = {'status': 'pass'}
except BaseException as error:
    report['error'] = type(error).__name__
print(json.dumps(report), flush=True)
"""


def check_sandbox():
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("bubblewrap is required; refusing unsandboxed code execution")
    if os.geteuid() == 0:
        raise RuntimeError("run code grading as a non-root user so the process-count limit applies")
    return executable


def run_python(code, test, entry_point, *, seconds=15):
    """Run a candidate against a frozen `check(entry_point)` test."""
    return _run(dict(code=code, test=test, entry_point=entry_point), seconds)


def _run(job, seconds):
    """Expose only read-only Python/runtime files; no host home, network, or corpus."""
    executable = check_sandbox()
    if not 0 < seconds <= 60:
        raise ValueError("code timeout must be in (0, 60] seconds")
    with tempfile.TemporaryDirectory(prefix="speck-code-") as temporary:
        root = Path(temporary)
        (root / "job.json").write_text(json.dumps(job))
        (root / "runner.py").write_text(WORKER)
        command = [
            executable,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for name in ("/bin", "/lib", "/lib64"):
            if Path(name).is_symlink():
                command += ["--symlink", os.readlink(name), name]
            elif Path(name).exists():
                command += ["--ro-bind", name, name]
        for prefix in sorted({sys.prefix, sys.base_prefix}):
            if prefix != "/usr" and not prefix.startswith("/usr/"):
                command += ["--ro-bind", prefix, prefix]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--ro-bind",
            str(root / "job.json"),
            "/job.json",
            "--ro-bind",
            str(root / "runner.py"),
            "/runner.py",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "1",
            "--setenv",
            "OMP_NUM_THREADS",
            "1",
            sys.executable,
            "-I",
            "/runner.py",
        ]
        started = time.monotonic()
        with (root / "output").open("w+b") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=output, start_new_session=True
            )
            timed_out = False
            try:
                process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            output.seek(0)
            report_bytes = output.read(8192)
        if timed_out:
            result = {"status": "timeout"}
        elif process.returncode != 0:
            result = {
                "status": "runner_failure",
                "returncode": process.returncode,
                "error": report_bytes.decode(errors="replace")[:2000],
            }
        else:
            try:
                result = json.loads(report_bytes)
                if result.get("status") not in {"pass", "failed"}:
                    raise ValueError("invalid worker status")
            except (ValueError, AttributeError):
                result = {"status": "runner_failure", "error": "missing or invalid worker result"}
        return {**result, "wall_seconds": time.monotonic() - started}
