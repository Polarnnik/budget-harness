"""Entry point: python /app/run_agent.py --tasks-dir /data/tasks

Environment: LLM_BASE_URL, LLM_API_KEY, BUDGET (output-token equivalents).
Optional: HARNESS_TRACE, path of the per-call trace (default /tmp/harness/trace.jsonl).
"""
import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

STARTED = time.monotonic()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    args = ap.parse_args()
    tasks_dir = Path(args.tasks_dir).resolve()

    # The image runs read-only: anything a library writes to the current
    # directory must land in /tmp. /app stays importable through sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    scratch = Path(tempfile.gettempdir()) / "harness"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from harness import agent, config
    from harness.llm import LLM, Budget

    try:
        total = float(os.environ["BUDGET"])
    except (KeyError, ValueError):
        total = config.DEFAULT_BUDGET
    budget = Budget(total)
    trace = Path(os.environ.get("HARNESS_TRACE") or scratch / "trace.jsonl")
    llm = LLM(os.environ.get("LLM_BASE_URL", ""), os.environ.get("LLM_API_KEY", ""), budget, trace)

    status = asyncio.run(agent.run(tasks_dir, llm, STARTED + config.RUN_DEADLINE_S))
    done = sum(s == "answered" for s in status.values())
    logging.getLogger("harness").info(
        "finished: %d/%d answered, %d calls, %.0f of %.0f tokens, %.0f s",
        done, len(status), llm.calls, budget.spent, budget.total, time.monotonic() - STARTED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
