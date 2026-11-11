"""Scheduler: solve tasks in order of reward per expected token, then spend spare budget on second opinions."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from . import config
from .sandbox import drop_workdir, make_workdir
from .solvers import coding, pot
from .tasks import Task, infer_type, load_tasks, missing_input, write_answer

log = logging.getLogger("harness")


def prompt_for(task: Task, kind: str) -> tuple[str, str, str]:
    """(context, prelude, shot) of a task that is solved by a generated program."""
    if kind == "data_analysis" and (table := pot.data_context(task)):
        return *table, kind
    if kind == "search":
        if cards := pot.search_context(task):
            return *cards, kind
        if notes := pot.notes_context(task):
            return *notes, "notes"
    return pot.generic_context(task), "", "math" if kind == "math" else ""


async def solve_task(task: Task, llm, chain=None, label: str = "", note: str = "") -> pot.Solution:
    if missing_input(task):
        log.info("%s: no input files for a %s task, skipped", task.id, task.type)
        return pot.Solution()
    kind = infer_type(task)
    work = make_workdir(task)
    try:
        if kind == "coding":
            return pot.Solution(await coding.solve(task, work, llm))
        return await pot.solve(task, work, llm, *prompt_for(task, kind), chain=chain, label=label, note=note)
    finally:
        drop_workdir(work)


def plan(tasks: list[Task]) -> list[Task]:
    """Greedy knapsack order: most reward per expected token first."""
    def value(t: Task) -> float:
        return t.reward / config.EXPECTED_COST.get(infer_type(t), config.DEFAULT_EXPECTED_COST)
    return sorted(tasks, key=value, reverse=True)


async def run(tasks_dir: Path, llm, deadline: float) -> dict[str, str]:
    """Solve what the budget and the clock allow; each answer is saved at once."""
    tasks = plan(load_tasks(tasks_dir))
    log.info("%d tasks, budget %.0f", len(tasks), llm.budget.total)
    status: dict[str, str] = {}
    solved: dict[str, pot.Solution] = {}
    gate = asyncio.Semaphore(config.MAX_PARALLEL)

    async def one(task: Task) -> None:
        async with gate:
            if time.monotonic() > deadline:
                status[task.id] = "skipped: out of time"
                return
            spent = llm.budget.spent
            try:
                solution = await solve_task(task, llm)
            except Exception as e:  # noqa: BLE001 - one bad task must not stop the run
                log.exception("%s crashed", task.id)
                status[task.id] = f"error: {e}"
                return
            if solution.answer is None:
                status[task.id] = "no answer"
            elif write_answer(task, solution.answer):
                status[task.id], solved[task.id] = "answered", solution
            else:
                status[task.id] = "write failed"
            log.info("%s (%s, reward %g): %s, ledger %.0f -> %.0f", task.id, task.type,
                     task.reward, status[task.id], spent, llm.budget.spent)

    async def second_opinion(task: Task) -> None:
        """Solve an answered task again from scratch; settle a disagreement with both programs in view."""
        async with gate:
            if time.monotonic() > deadline or llm.budget.spent > config.VERIFY_BUDGET_SHARE * llm.budget.total:
                return
            first = solved[task.id]
            try:
                second = await solve_task(task, llm, [config.VERIFY_STEP], "v")
                if second.answer is None or second.answer == first.answer:
                    return
                note = pot.DISPUTE.format(a=first.answer, code_a=first.code, b=second.answer, code_b=second.code)
                final = (await solve_task(task, llm, [config.VERIFY_STEP], "d", note)).answer
            except Exception:  # noqa: BLE001 - the first answer is already on disk
                log.exception("%s: second opinion crashed", task.id)
                return
            log.info("%s: second opinion %r against %r, settled as %r", task.id, second.answer, first.answer, final)
            if final is not None and final != first.answer and write_answer(task, final):
                status[task.id] = "answered, changed by second opinion"

    await asyncio.gather(*(one(t) for t in tasks))
    if config.VERIFY:
        # Every task has its first answer on disk by now. Dearest tasks go first,
        # in case the budget share for second opinions runs out.
        doubtful = [t for t in tasks if t.id in solved and solved[t.id].risky]
        await asyncio.gather(*(second_opinion(t) for t in sorted(doubtful, key=lambda t: -t.reward)))
    return status
