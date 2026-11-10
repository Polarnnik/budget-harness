"""Run untrusted code (generated programs, task tests) in a throwaway copy under /tmp."""
from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task

RUNNER = Path(__file__).with_name("runner.py")
MAX_OUTPUT = 20_000
SCRATCH = Path(tempfile.gettempdir()) / "harness"


@dataclass
class ExecResult:
    code: int            # process exit code; -1 on timeout
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def error(self) -> str:
        """Last line of the traceback, enough for a retry prompt."""
        if self.code == -1:
            return "TimeoutError: the program ran too long"
        lines = [l for l in self.err.strip().splitlines() if l.strip()]
        return lines[-1][:300] if lines else f"exit code {self.code}"


def make_workdir(task: Task) -> Path:
    """Fresh copy of the task workspace. Task files never run in place."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{task.dir.name}-", dir=SCRATCH))
    work = root / "workspace"
    if task.workspace.is_dir():
        shutil.copytree(task.workspace, work, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
    else:
        work.mkdir()
    return work


def drop_workdir(work: Path) -> None:
    shutil.rmtree(work.parent, ignore_errors=True)


def _limits(timeout: float):
    def apply():
        cpu = int(timeout) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    return apply


def _env() -> dict[str, str]:
    # The proxy key and URL stay out of the child: task code has no use for them.
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


async def run(argv: list[str], cwd: Path, timeout: float) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, preexec_fn=_limits(timeout),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        code = proc.returncode
    except asyncio.TimeoutError:
        code, out, err = -1, b"", b""
    finally:
        # Kill the whole session: tests may leave children behind.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if proc.returncode is None:
            await proc.wait()
    return ExecResult(code, out.decode("utf-8", "replace")[:MAX_OUTPUT],
                      err.decode("utf-8", "replace")[-MAX_OUTPUT:])


async def run_program(code: str, work: Path, prelude: str = "", timeout: float = 20) -> ExecResult:
    """Execute a generated program; stdout is its answer (see runner.py)."""
    aux = work.parent
    (aux / "program.py").write_text(code, encoding="utf-8")
    (aux / "prelude.py").write_text(prelude, encoding="utf-8")
    # -I keeps the workspace off sys.path: a planted csv.py must not shadow the stdlib.
    return await run([sys.executable, "-I", "-B", str(RUNNER),
                      str(aux / "program.py"), str(aux / "prelude.py")], work, timeout)


async def run_tests(work: Path, test_files: list[Path], timeout: float = 30) -> ExecResult:
    """Run each test file the way the instruction does: `python3 test_x.py`."""
    out, err = [], []
    for t in test_files:
        r = await run([sys.executable, "-B", t.name], work / t.parent, timeout)
        out.append(r.out)
        err.append(r.err)
        if not r.ok:
            return ExecResult(r.code, "".join(out), "".join(err))
    return ExecResult(0, "".join(out), "".join(err))
