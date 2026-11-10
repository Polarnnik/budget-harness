"""Scheduler: order tasks by reward per expected token and solve them in parallel."""
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


async def solve_task(task: Task, llm) -> str | None:
    if missing_input(task):
        log.info("%s: no input files for a %s task, skipped", task.id, task.type)
        return None
    kind = infer_type(task)
    work = make_workdir(task)
    try:
        if kind == "coding":
            return await coding.solve(task, work, llm)
        family = None
        if kind == "data_analysis":
            family = pot.data_context(task)
        elif kind == "search":
            family = pot.search_context(task)
        if family:
            return await pot.solve(task, work, llm, *family, shot=kind)
        return await pot.solve(task, work, llm, pot.generic_context(task), shot="math" if kind == "math" else "")
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
    gate = asyncio.Semaphore(config.MAX_PARALLEL)

    async def one(task: Task) -> None:
        async with gate:
            if time.monotonic() > deadline:
                status[task.id] = "skipped: out of time"
                return
            spent = llm.budget.spent
            try:
                answer = await solve_task(task, llm)
            except Exception as e:  # noqa: BLE001 - one bad task must not stop the run
                log.exception("%s crashed", task.id)
                status[task.id] = f"error: {e}"
                return
            if answer is None:
                status[task.id] = "no answer"
            else:
                status[task.id] = "answered" if write_answer(task, answer) else "write failed"
            log.info("%s (%s, reward %g): %s, ledger %.0f -> %.0f", task.id, task.type,
                     task.reward, status[task.id], spent, llm.budget.spent)

    await asyncio.gather(*(one(t) for t in tasks))
    return status
