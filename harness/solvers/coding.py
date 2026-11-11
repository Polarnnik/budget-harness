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
    "You fix bugs in Python code. Source lines are numbered. Hidden tests are wider than "
    "the shown ones, so a passing test proves little: judge every function by what its "
    "name promises on ordinary inputs (duplicates, mixed case, unsorted data). Functions "
    "may assume valid, non-empty input: missing checks are not bugs. "
    "Go through the functions one by one. For each write `# name: ok` or `# name: bug`, "
    "and under a bug its corrected lines as `N| code`, with the same number and "
    "indentation as shown. To drop a line reply `N|`. With several source files, put "
    "`=== file.py ===` before its lines. Change only lines that are wrong: no new "
    "lines, no restyling. Never change the tests. No other text."
)

EDIT = re.compile(r"^\s*(?:L|line\s*)?(\d+)\s*[:|] ?(.*)$", re.I)
HEADER = re.compile(r"^\s*[#=]+\s*(?:file:?\s*)?(\S+\.py)\s*=*\s*$", re.I)
MAX_SOURCE_CHARS = 12_000


def numbered(text: str) -> str:
    return "\n".join(f"{i}| {line}" for i, line in enumerate(text.splitlines(), 1))


def parse_edits(reply: str, default_file: str, truncated: bool = False) -> dict[str, dict[int, str]]:
    """Reply -> {file: {line number: new text}}. A reply cut off by max_tokens loses its last line."""
    edits: dict[str, dict[int, str]] = {}
    current = default_file
    for line in reply.splitlines()[:-1 if truncated else None]:
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
    """No fix to the sources can make these tests pass.

    Either two assertEqual calls expect different literals from the same call,
    or the test file itself is not valid Python.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return True         # a test file that does not parse cannot pass, and it must not be edited
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


def done_word(instruction: str) -> str:
    """The word the task wants in answer.txt once the tests pass, e.g. "the word DONE"."""
    m = re.search(r"\bword\s+[`'\"]?([A-Za-z]+)", instruction)
    return m.group(1) if m else "DONE"


async def solve(task: Task, work: Path, llm) -> str | None:
    files = task.files()
    tests = [p for p in files if _is_test(p)]
    sources = [p for p in files if p.suffix == ".py" and not _is_test(p)]
    original = {p: (work / p).read_text(encoding="utf-8", errors="replace") for p in sources}
    if any(contradictory((work / p).read_text(encoding="utf-8", errors="replace")) for p in tests):
        log.info("%s: tests cannot pass whatever the sources are, unsolvable", task.id)
        return None
    # Operator swaps undo the common injected bugs without the model. The model
    # still reads the result: a bug can sit behind a test that happens to pass.
    if len(sources) == 1 and await mutate.repair(work, sources[0], tests, config.TEST_TIMEOUT_S):
        log.info("%s: failing tests repaired by operator swap", task.id)
        original = {p: (work / p).read_text(encoding="utf-8") for p in sources}

    async def check() -> tuple[set[str], str]:
        r = await run_tests(work, tests, config.TEST_TIMEOUT_S)
        if r.ok:
            return set(), ""
        return set(mutate.FAILED.findall(r.err + r.out)) or {"<crash>"}, failure_digest(r.err + r.out, work)

    # `state` is the best version so far. Every step shows it to the model with
    # fresh line numbers, so edits never refer to a numbering the model lost.
    state = original
    failing, digest = await check()
    was_green = not failing
    hints = ""
    if len(sources) == 1:
        open_lines = await mutate.undecided(work, sources[0], tests)
        if open_lines:
            lines = state[sources[0]].splitlines()
            hints = ("\nThe tests give the same result with either version of the lines below, so they prove "
                     "nothing there. Decide by the function's name; to switch a line, reply with its `or` version.\n"
                     + "\n".join(f"{n}| {lines[n - 1]}\nor| {other}" for n, other in open_lines))
    read_only = "\n\n".join(f"=== {p.as_posix()} (tests, read-only) ===\n"
                            f"{(work / p).read_text(encoding='utf-8', errors='replace')}" for p in tests)
    for step, (model, max_tokens) in enumerate(config.CHAIN):
        if step and not failing:
            break
        listing = "\n\n".join(f"=== {p.as_posix()} ===\n{numbered(state[p])}" for p in sources)
        status = ("All shown tests pass." if not failing else f"Failing now:\n{digest}") + (hints if step == 0 else "")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{listing}\n\n{read_only}"[:MAX_SOURCE_CHARS] + f"\n\n{status}"}]
        reply = await llm.ask(model, messages, max_tokens, tag=f"{task.id}#{step}")
        if reply is None:
            break
        edits = parse_edits(reply.text, sources[0].as_posix(), reply.truncated)
        attempt = {p: apply_edits(state[p], edits.get(p.as_posix(), {})) for p in sources}
        for p in sources:
            (work / p).write_text(attempt[p], encoding="utf-8")
        now, now_digest = await check()
        # Keep an attempt that is green, or that fixed some tests without breaking the module.
        if not now or (failing and "<crash>" not in now and len(now) < len(failing)):
            state, failing, digest = attempt, now, now_digest
        else:
            log.info("%s step %d: edits dropped, tests got no better", task.id, step)
        if was_green:       # a review that breaks passing code is dropped, not argued with
            break

    for p in sources:
        (work / p).write_text(state[p], encoding="utf-8")
    if failing:
        return None         # tests still fail: no answer file, the sources stay as given
    _copy_back(task, work, sources)
    return done_word(task.instruction)
