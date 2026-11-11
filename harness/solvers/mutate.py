"""Token-free repair of single-operator bugs.

The example tasks inject bugs by swapping one operator or one builtin
(`+=`/`-=`, `<`/`>`, `sorted`/`reversed`, `% 2 == 0`/`== 1`). Undoing such a
swap needs no model: try every inverse swap, keep the ones the task's tests
accept. The repair is used only when it is unambiguous; anything else goes to
the model.
"""
from __future__ import annotations

import ast
import re
import time
from pathlib import Path

from ..sandbox import run_tests

BINARY = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div, ast.FloorDiv],
          ast.Div: [ast.Mult], ast.FloorDiv: [ast.Mult]}
COMPARE = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.NotEq,
           ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BOOLEAN = {ast.And: ast.Or, ast.Or: ast.And}
NAMES = {"sorted": "reversed", "reversed": "sorted", "min": "max", "max": "min", "any": "all", "all": "any"}
UNITS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)     # top-level pieces that get re-printed
MUTANT_TIMEOUT_S = 3
SLOW_TESTS_S = 2          # a suite slower than this is not re-run once per swap
ANALYSIS_BUDGET_S = 60    # wall-clock cap for trying swaps on one task
MAX_MUTANTS = 150
FAILED = re.compile(r"^(?:FAIL|ERROR): (\S+)", re.M)


def _sites(func: ast.AST):
    """Yield (object, attribute or index, replacement, line number) for every swap inside a function."""
    for node in ast.walk(func):
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            for op in BINARY.get(type(node.op), []):
                yield node, "op", op(), node.lineno
        elif isinstance(node, ast.BoolOp):
            yield node, "op", BOOLEAN[type(node.op)](), node.lineno
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in COMPARE:
                    yield node.ops, i, COMPARE[type(op)](), node.lineno
            parity = isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mod)
            for i, c in enumerate(node.comparators):
                if parity and isinstance(c, ast.Constant) and c.value in (0, 1) and type(c.value) is int:
                    yield node.comparators, i, ast.Constant(1 - c.value), node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NAMES:
            yield node.func, "id", NAMES[node.func.id], node.lineno
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and node.slice.step is None:
            s = node.slice
            if (s.lower is None) != (s.upper is None):     # xs[:n] <-> xs[n:]
                yield node, "slice", ast.Slice(lower=s.upper, upper=s.lower), node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield node, "value", not node.value, node.lineno


def _swap(target, key, value):
    """Set target[key] or target.key; return the previous value."""
    if isinstance(target, list):
        old, target[key] = target[key], value
    else:
        old = getattr(target, key)
        setattr(target, key, value)
    return old


def _render(lines: list[str], funcs: list[ast.AST]) -> str:
    """Original text with the given functions re-printed from their (mutated) AST."""
    out = list(lines)
    for f in sorted({id(f): f for f in funcs}.values(), key=lambda f: f.lineno, reverse=True):
        start = min([f.lineno] + [d.lineno for d in f.decorator_list]) - 1
        out[start:f.end_lineno] = ast.unparse(f).splitlines()
    return "\n".join(out) + "\n"


async def _baseline(work: Path, tests: list[Path], timeout: float) -> set[str] | None:
    """Failing tests of the unchanged source, or None when swaps cannot be tried.

    Every swap costs one test run. A suite that hangs or is slow by design
    would turn that into minutes per task, so such tasks skip the analysis.
    """
    started = time.monotonic()
    base = await _failing(work, tests, timeout)
    if "<crash>" in base or time.monotonic() - started > SLOW_TESTS_S:
        return None
    return base


async def _failing(work: Path, tests: list[Path], timeout: float) -> set[str]:
    r = await run_tests(work, tests, timeout)
    if r.ok:
        return set()
    return set(FAILED.findall(r.err + r.out)) or {"<crash>"}


async def repair(work: Path, source: Path, tests: list[Path], timeout: float) -> bool:
    """Fix `source` in `work` with inverse operator swaps. True only for an unambiguous repair."""
    path = work / source
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    funcs = [n for n in tree.body if isinstance(n, UNITS)]
    test_text = "\n".join((work / t).read_text(encoding="utf-8", errors="replace") for t in tests)
    # A function no test mentions may hide a bug only a reader can see: leave the task to the model.
    if not funcs or any(not re.search(rf"\b{re.escape(f.name)}\b", test_text) for f in funcs):
        return False
    base = await _baseline(work, tests, timeout)
    if not base:
        return False
    deadline = time.monotonic() + ANALYSIS_BUDGET_S

    lines = text.splitlines()
    mutants = [(f, site) for f in funcs for site in _sites(f)][:MAX_MUTANTS]
    fixes: dict[str, list[int]] = {t: [] for t in base}
    try:
        for i, (func, (target, key, value, _)) in enumerate(mutants):
            if time.monotonic() > deadline:
                return False
            old = _swap(target, key, value)
            path.write_text(_render(lines, [func]), encoding="utf-8")
            _swap(target, key, old)
            left = await _failing(work, tests, MUTANT_TIMEOUT_S)
            if left < base:                                  # fixes something, breaks nothing
                for t in base - left:
                    fixes[t].append(i)
        chosen = set()
        for t, found in fixes.items():
            # Each failing test must point at swaps inside one function; several
            # swaps there are equivalent readings of one bug (`!= 1` vs `== 0`).
            if not found or len({mutants[i][0].name for i in found}) != 1:
                return False
            chosen.add(found[0])
        for i in chosen:
            _swap(*mutants[i][1][:3])
        path.write_text(_render(lines, [mutants[i][0] for i in chosen]), encoding="utf-8")
        if not await _failing(work, tests, timeout):
            return True
        return False
    finally:
        if await _failing(work, tests, timeout):             # anything short of green: restore
            path.write_text(text, encoding="utf-8")


async def undecided(work: Path, source: Path, tests: list[Path], limit: int = 12) -> list[tuple[int, str]]:
    """Swaps that the tests cannot tell from the current code: (line number, other version).

    If the same tests pass and fail with `sorted` as with `reversed`, they say
    nothing about that line and a bug may hide there. These lines are the only
    places where a swapped operator can survive the visible tests, so they are
    what the reviewer must judge.
    """
    path = work / source
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    found: list[tuple[int, str]] = []
    base = await _baseline(work, tests, MUTANT_TIMEOUT_S)
    if base is None:            # nothing runs, or too slow: no verdict on any line
        return []
    deadline = time.monotonic() + ANALYSIS_BUDGET_S
    try:
        for func in (n for n in tree.body if isinstance(n, UNITS)):
            heads = {}
            for stmt in ast.walk(func):      # innermost statement that starts on each line
                if isinstance(stmt, ast.stmt):
                    heads[stmt.lineno] = stmt
            for target, key, value, line in list(_sites(func))[:MAX_MUTANTS]:
                if line not in heads or len(found) >= limit or time.monotonic() > deadline:
                    continue
                current = lines[line - 1]
                old = _swap(target, key, value)
                other = current[: len(current) - len(current.lstrip())] + ast.unparse(heads[line]).splitlines()[0]
                path.write_text(_render(lines, [func]), encoding="utf-8")
                _swap(target, key, old)
                if other.strip() != current.strip() and await _failing(work, tests, MUTANT_TIMEOUT_S) == base:
                    found.append((line, other))
    finally:
        path.write_text(text, encoding="utf-8")
    return found
