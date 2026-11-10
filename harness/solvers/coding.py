"""Bug-fixing tasks: the model returns only the corrected lines.

The source goes to the model with line numbers and the answer comes back as
`N: new code`. A one-line fix costs about ten output tokens instead of a
rewritten file. The patch is applied to a copy under /tmp, the task's own
tests run there, and only source files that pass are copied back.
"""
from __future__ import annotations

import ast
import logging
import re
import shutil
from pathlib import Path

from .. import config
from ..sandbox import run_tests
from . import mutate
from ..tasks import Task, _is_test

log = logging.getLogger("harness")

SYSTEM = (
    "You fix bugs in Python code. Source lines are numbered. Check every function "
    "against its name, its docstring and the tests; hidden tests also cover functions "
    "the shown tests miss. Never change the tests. Reply only with corrected lines as "
    "`N: code`, one per line, keeping the indentation. To drop a line reply `N:`. "
    "With several source files, put `# file.py` before its lines. Change only lines that "
    "are wrong: add no input checks, no restyling. If nothing needs fixing reply OK. "
    "No other text."
)
RETRY = "Tests still fail after your edits:\n{error}\nReply with corrected lines for the original numbering."

EDIT = re.compile(r"^\s*(?:L|line\s*)?(\d+)\s*[:|] ?(.*)$", re.I)
HEADER = re.compile(r"^\s*#+\s*(?:file:?\s*)?(\S+\.py)\s*$", re.I)
MAX_SOURCE_CHARS = 12_000


def numbered(text: str) -> str:
    return "\n".join(f"{i}: {line}" for i, line in enumerate(text.splitlines(), 1))


def parse_edits(reply: str, default_file: str) -> dict[str, dict[int, str]]:
    """Reply -> {file: {line number: new text}}."""
    edits: dict[str, dict[int, str]] = {}
    current = default_file
    for line in reply.splitlines():
        if line.strip().startswith("```"):
            continue
        h = HEADER.match(line)
        if h:
            current = h.group(1)
            continue
        m = EDIT.match(line)
        if m:
            edits.setdefault(current, {})[int(m.group(1))] = m.group(2).rstrip()
    return edits


def apply_edits(text: str, edits: dict[int, str]) -> str:
    lines = text.splitlines()
    for n, new in edits.items():
        if not 1 <= n <= len(lines):
            continue
        old = lines[n - 1]
        if new and not new[0].isspace() and old[:1].isspace():
            # The model dropped the indentation; the line keeps its old one.
            new = old[: len(old) - len(old.lstrip())] + new
        lines[n - 1] = new if new.strip() else None
    return "\n".join(l for l in lines if l is not None) + "\n"


def failure_digest(out: str, work: Path, limit: int = 1200) -> str:
    """Assertion lines and the summary: what the model needs from a unittest run."""
    out = out.replace(f"{work}/", "")
    keep = [l for l in out.splitlines()
            if re.match(r"\s*(FAIL|ERROR|\w*Error|AssertionError|Ran |FAILED|File )", l)]
    return "\n".join(keep or out.strip().splitlines()[-15:])[-limit:]


def _copy_back(task: Task, work: Path, sources: list[Path]) -> None:
    """Sources only, never tests, and only a state the visible tests accept."""
    for p in sources:
        try:
            shutil.copyfile(work / p, task.workspace / p)
        except OSError as e:
            log.error("%s: cannot write %s: %s", task.id, p, e)


def contradictory(test_source: str) -> bool:
    """Two assertEqual calls expect different literals from the same call: no fix can pass."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return False
    seen: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "assertEqual"
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Call)):
            try:
                expected = repr(ast.literal_eval(node.args[1]))
            except ValueError:
                continue
            if seen.setdefault(ast.dump(node.args[0]), expected) != expected:
                return True
    return False


async def solve(task: Task, work: Path, llm) -> str | None:
    files = task.files()
    tests = [p for p in files if _is_test(p)]
    sources = [p for p in files if p.suffix == ".py" and not _is_test(p)]
    original = {p: (work / p).read_text(encoding="utf-8", errors="replace") for p in sources}
    if any(contradictory((work / p).read_text(encoding="utf-8", errors="replace")) for p in tests):
        log.info("%s: tests contradict each other, unsolvable", task.id)
        return None
    # Operator swaps undo the common injected bugs without the model. The model
    # still reads the result: a bug can sit behind a test that happens to pass.
    if len(sources) == 1 and await mutate.repair(work, sources[0], tests, config.TEST_TIMEOUT_S):
        log.info("%s: failing tests repaired by operator swap", task.id)
        original = {p: (work / p).read_text(encoding="utf-8") for p in sources}

    parts = []
    for p in sources:
        parts.append(f"# {p.as_posix()}\n{numbered(original[p])}")
    for p in tests:
        parts.append(f"# {p.as_posix()} (tests, read-only)\n{(work / p).read_text(encoding='utf-8', errors='replace')}")
    listing = "\n\n".join(parts)[:MAX_SOURCE_CHARS]
    before = await run_tests(work, tests, config.TEST_TIMEOUT_S)
    status = "All shown tests pass." if before.ok else f"Failing now:\n{failure_digest(before.err + before.out, work)}"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{listing}\n\n{status}"}]

    solved = False
    edits: dict[str, dict[int, str]] = {}
    for step, (model, max_tokens) in enumerate(config.CHAIN):
        reply = await llm.ask(model, messages, max_tokens, tag=f"{task.id}#{step}")
        if reply is None:
            break
        # A follow-up reply often repeats only the lines it changes, so edits accumulate.
        for name, lines in parse_edits(reply.text, sources[0].as_posix()).items():
            edits.setdefault(name, {}).update(lines)
        for p in sources:   # edits always apply to the original numbering
            (work / p).write_text(apply_edits(original[p], edits.get(p.as_posix(), {})), encoding="utf-8")
        after = await run_tests(work, tests, config.TEST_TIMEOUT_S)
        if after.ok:
            solved = True
            break
        log.info("%s step %d: tests still fail", task.id, step)
        if before.ok:       # a review that breaks passing code is dropped, not argued with
            break
        messages += [{"role": "assistant", "content": reply.text},
                     {"role": "user", "content": RETRY.format(error=failure_digest(after.err + after.out, work))}]

    if not (solved or before.ok):
        return None         # tests still fail: no answer file, the sources stay as given
    if not solved:
        for p in sources:
            (work / p).write_text(original[p], encoding="utf-8")
    _copy_back(task, work, sources)
    return "DONE"
