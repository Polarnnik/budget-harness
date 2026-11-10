"""Task discovery and answer writing."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("harness")

FAMILIES = ("math", "data_analysis", "search", "coding")
DATA_EXT = {".csv", ".tsv"}
TEXT_EXT = {".md", ".txt"}


@dataclass
class Task:
    id: str
    dir: Path
    type: str
    instruction: str
    reward: float
    check: str

    @property
    def workspace(self) -> Path:
        return self.dir / "workspace"

    def files(self) -> list[Path]:
        """Workspace files, relative paths, sorted; answer.txt excluded."""
        ws = self.workspace
        if not ws.is_dir():
            return []
        return sorted(
            p.relative_to(ws) for p in ws.rglob("*")
            if p.is_file() and p.name != "answer.txt" and "__pycache__" not in p.parts
        )


def _is_test(p: Path) -> bool:
    return p.suffix == ".py" and (p.name.startswith("test") or p.stem.endswith("_test"))


def _fits(kind: str, files: list[Path]) -> bool:
    """Does the workspace hold the input files this task family needs?"""
    py = [p for p in files if p.suffix == ".py"]
    if kind == "coding":
        return any(_is_test(p) for p in py) and any(not _is_test(p) for p in py)
    if kind == "data_analysis":
        return any(p.suffix.lower() in DATA_EXT for p in files)
    if kind == "search":
        return any(p.suffix.lower() in TEXT_EXT for p in files)
    return kind == "math"


def infer_type(task: Task) -> str:
    """Solver family: the declared task_type, or a guess from the files when it is unknown."""
    if task.type in FAMILIES:
        return task.type
    files = task.files()
    return next((k for k in ("coding", "data_analysis", "search") if _fits(k, files)),
                "generic" if files else "math")


def missing_input(task: Task) -> bool:
    """The declared family has nothing to work on: no table, no cards, no code.

    Such a task cannot be solved from the data given, so it gets no model call
    and no answer file.
    """
    return task.type in FAMILIES and not _fits(task.type, task.files())


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks = []
    for d in sorted(Path(tasks_dir).iterdir()):
        spec = d / "task.json"
        if d.name.startswith((".", "__")) or not spec.is_file():
            continue
        try:
            raw = json.loads(spec.read_text(encoding="utf-8", errors="replace"))
            reward = float(raw.get("reward", 1))
        except (ValueError, TypeError, AttributeError) as e:
            log.warning("skip %s: bad task.json (%s)", d.name, e)
            continue
        tasks.append(Task(
            id=str(raw.get("task_id") or d.name),
            dir=d,
            type=str(raw.get("task_type") or ""),
            instruction=str(raw.get("instruction") or ""),
            reward=reward,
            check=str(raw.get("check") or ""),
        ))
    return tasks


def write_answer(task: Task, text: str) -> bool:
    """Write answer.txt next to task.json, as the organisers require.

    Instructions say "current directory", which is the workspace, so an
    existing workspace/ gets a copy; the directory is never created for it.
    """
    ok = False
    for d in (task.dir, task.workspace):
        if not d.is_dir():
            continue
        try:
            (d / "answer.txt").write_text(text.strip() + "\n", encoding="utf-8")
            ok = ok or d == task.dir
        except OSError as e:
            log.error("cannot write answer for %s in %s: %s", task.id, d, e)
    return ok
