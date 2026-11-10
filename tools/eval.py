"""Run the harness on a copy of a task set and score it against `expected`.

    LLM_BASE_URL=... LLM_API_KEY=... python tools/eval.py examples/tasks --budget 5000 --out runs/demo

The task set is copied first, so the originals keep their buggy sources and
stay reusable. Scoring mirrors the organisers' checks: exact_int, exact_str,
and unittest (the visible tests, plus hidden_tests/ when a generated task has them).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def passed(task_dir: Path, spec: dict) -> bool:
    answer_file = task_dir / "answer.txt"
    if not answer_file.is_file():
        return False
    answer, expected = answer_file.read_text().strip(), str(spec.get("expected", "")).strip()
    if spec.get("check") == "unittest":
        ws = task_dir / "workspace"
        hidden = task_dir / "hidden_tests"
        if hidden.is_dir():
            shutil.copytree(hidden, ws, dirs_exist_ok=True)
        tests = sorted(p.name for p in ws.glob("test*.py"))
        return answer == "DONE" and bool(tests) and all(
            subprocess.run([sys.executable, "-B", t], cwd=ws, capture_output=True, timeout=60).returncode == 0
            for t in tests)
    if spec.get("check") == "exact_int":
        try:
            return int(answer) == int(expected)
        except ValueError:
            return False
    return answer == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--budget", default="100000")
    ap.add_argument("--out", default="", help="run directory (default runs/<timestamp>)")
    args = ap.parse_args()

    out = Path(args.out or ROOT / "runs" / time.strftime("%m%d-%H%M%S")).resolve()
    tasks = out / "tasks"
    shutil.copytree(args.tasks, tasks)
    for stale in tasks.rglob("answer.txt"):
        stale.unlink()
    env = dict(os.environ, BUDGET=args.budget, HARNESS_TRACE=str(out / "trace.jsonl"))
    started = time.monotonic()
    with open(out / "run.log", "w") as log:
        subprocess.run([sys.executable, "-B", str(ROOT / "run_agent.py"), "--tasks-dir", str(tasks)],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - started

    cost = defaultdict(float)
    calls = defaultdict(int)
    if (out / "trace.jsonl").is_file():
        for line in (out / "trace.jsonl").read_text().splitlines():
            row = json.loads(line)
            task_id = row.get("tag", "").split("#")[0]
            cost[task_id] += row.get("cost", 0)
            calls[task_id] += "cost" in row

    by_type = defaultdict(lambda: [0, 0, 0.0, 0.0])   # solved, total, reward, cost
    print(f"{'task':28} {'ok':3} {'reward':>6} {'calls':>5} {'cost':>7}")
    for d in sorted(tasks.iterdir()):
        if not (d / "task.json").is_file():
            continue
        spec = json.loads((d / "task.json").read_text())
        ok = passed(d, spec)
        row = by_type[spec.get("task_type", "?")]
        row[0] += ok
        row[1] += 1
        row[2] += spec.get("reward", 0) * ok
        row[3] += cost[spec["task_id"]]
        print(f"{d.name:28} {'+' if ok else '-':3} {spec.get('reward', 0):6g} "
              f"{calls[spec['task_id']]:5d} {cost[spec['task_id']]:7.1f}")
    print()
    for kind, (ok, n, reward, c) in sorted(by_type.items()):
        print(f"{kind:16} {ok}/{n} solved, reward {reward:g}, cost {c:.0f} ({c / n:.0f} per task)")
    total = [sum(r[i] for r in by_type.values()) for i in range(4)]
    print(f"{'TOTAL':16} {total[0]}/{total[1]} solved, reward {total[2]:g}, cost {total[3]:.0f}, {seconds:.0f} s")
    print(f"run directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
